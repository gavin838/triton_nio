#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;

namespace {
struct SplatOpConversion : public ConvertOpToLLVMPattern<triton::SplatOp> {
  using ConvertOpToLLVMPattern<triton::SplatOp>::ConvertOpToLLVMPattern;

  // Convert SplatOp or arith::ConstantOp with SplatElementsAttr to a
  // LLVM::StructType value.
  //
  // @elemType: the element type in operand.
  // @resType: the return type of the Splat-like op.
  // @constVal: a LLVM::ConstantOp or other scalar value.
  static Value convertSplatLikeOp(Type elemType, Type resType, Value constVal,
                                  const LLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
    auto tensorTy = resType.cast<RankedTensorType>();
    // Check the converted type for the tensor as depending on the encoding the
    // converter may pick different element types.
    auto srcType = typeConverter->convertType(tensorTy);
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(srcType))
      srcType = structTy.getBody()[0];
    // If the type sizes don't match we need to pack constants.
    if (srcType.isIntOrFloat() && constVal.getType().getIntOrFloatBitWidth() !=
                                      srcType.getIntOrFloatBitWidth()) {
      unsigned cstBitWidth = constVal.getType().getIntOrFloatBitWidth();
      unsigned srcBitWidth = srcType.getIntOrFloatBitWidth();
      assert(cstBitWidth <= srcBitWidth && srcBitWidth % cstBitWidth == 0);
      unsigned ratio = srcBitWidth / cstBitWidth;
      Type intTy = IntegerType::get(elemType.getContext(), cstBitWidth);
      VectorType vecType = VectorType::get(ratio, intTy);
      Value intCst = bitcast(constVal, intTy);
      Value vec = undef(vecType);
      for (unsigned i = 0; i < ratio; ++i)
        vec = insert_element(vecType, vec, intCst, int_val(32, i));
      constVal = vec;
    }
    auto llSrc = bitcast(constVal, srcType);
    size_t elemsPerThread = getTotalElemsPerThread(tensorTy);
    llvm::SmallVector<Value> elems(elemsPerThread, llSrc);
    return packLLElements(loc, typeConverter, elems, rewriter, resType);
  }

  LogicalResult matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto src = adaptor.getSrc();
    auto typeConverter = getTypeConverter();
    auto llStruct = convertSplatLikeOp(src.getType(), op.getType(), src,
                                       typeConverter, rewriter, loc);
    rewriter.replaceOp(op, {llStruct});
    return success();
  }
};

// This pattern helps to convert arith::ConstantOp(with SplatElementsAttr),
// the logic is the same as triton::SplatOp, so the underlying implementation
// is reused.
struct ArithConstantSplatOpConversion
    : public ConvertOpToLLVMPattern<arith::ConstantOp> {
  using ConvertOpToLLVMPattern<arith::ConstantOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    if (!value.dyn_cast<SplatElementsAttr>())
      return failure();

    auto loc = op->getLoc();

    LLVM::ConstantOp arithConstantOp;
    auto values = op.getValue().dyn_cast<SplatElementsAttr>();
    auto elemType = values.getElementType();

    Attribute val;
    if (elemType.isBF16() || type::isFloat(elemType)) {
      val = values.getValues<FloatAttr>()[0];
    } else if (type::isInt(elemType)) {
      val = values.getValues<IntegerAttr>()[0];
    } else {
      llvm::errs() << "ArithConstantSplatOpConversion get unsupported type: "
                   << value.getType() << "\n";
      return failure();
    }

    auto constOp = rewriter.create<LLVM::ConstantOp>(loc, elemType, val);
    auto typeConverter = getTypeConverter();
    auto llStruct = SplatOpConversion::convertSplatLikeOp(
        elemType, op.getType(), constOp, typeConverter, rewriter, loc);
    rewriter.replaceOp(op, llStruct);

    return success();
  }
};

struct CatOpConversion : public ConvertOpToLLVMPattern<CatOp> {
  using OpAdaptor = typename CatOp::Adaptor;

  explicit CatOpConversion(LLVMTypeConverter &typeConverter,

                           PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<CatOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType().template cast<RankedTensorType>();
    unsigned elems = getTotalElemsPerThread(resultTy);
    auto typeConverter = getTypeConverter();
    Type elemTy = typeConverter->convertType(resultTy.getElementType());
    SmallVector<Type> types(elems, elemTy);
    // unpack input values
    auto lhsVals = unpackLLElements(loc, adaptor.getLhs(), rewriter);
    auto rhsVals = unpackLLElements(loc, adaptor.getRhs(), rewriter);
    // concatenate (and potentially reorder) values
    SmallVector<Value> retVals;
    for (Value v : lhsVals)
      retVals.push_back(v);
    for (Value v : rhsVals)
      retVals.push_back(v);
    // pack and replace
    Value ret = packLLElements(loc, typeConverter, retVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct InterleaveOpConversion
    : public ConvertOpToLLVMPattern<ExperimentalInterleaveOp> {
  using OpAdaptor = typename ExperimentalInterleaveOp::Adaptor;

  explicit InterleaveOpConversion(LLVMTypeConverter &typeConverter,
                                  PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<ExperimentalInterleaveOp>(typeConverter,
                                                         benefit) {}

  LogicalResult
  matchAndRewrite(ExperimentalInterleaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We rely on the following invariants of this op (which are checked by its
    // verifier):
    //
    // - The op has a blocked encoding.
    // - The last dimension (the one we're interleaving) is also the most minor
    //   dimension.
    // - The input and output encodings are the same, except the output has
    //   twice as many elements per thread in the last dimension.
    //
    // With these invariants, interleaving is trivial: We just return the i'th
    // element from lhs, followed by the i'th elem from rhs.
    Location loc = op->getLoc();
    auto resultTy = op.getType().cast<RankedTensorType>();
    auto typeConverter = getTypeConverter();

    SmallVector<Value> lhsVals =
        unpackLLElements(loc, adaptor.getLhs(), rewriter);
    SmallVector<Value> rhsVals =
        unpackLLElements(loc, adaptor.getRhs(), rewriter);
    assert(lhsVals.size() == rhsVals.size());

    SmallVector<Value> interleavedVals;
    for (int i = 0; i < lhsVals.size(); i++) {
      interleavedVals.push_back(lhsVals[i]);
      interleavedVals.push_back(rhsVals[i]);
    }

    Value ret =
        packLLElements(loc, typeConverter, interleavedVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct ReshapeOpConversion : public ConvertOpToLLVMPattern<ReshapeOp> {
  using OpAdaptor = typename ReshapeOp::Adaptor;
  explicit ReshapeOpConversion(LLVMTypeConverter &typeConverter,

                               PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<ReshapeOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    if (triton::gpu::isExpensiveView(op.getSrc().getType(), op.getType())) {
      return emitOptionalError(loc,
                               "expensive view not supported on reshape op");
    }
    auto resultTy = op.getType().template cast<RankedTensorType>();
    auto srcTy = op.getSrc().getType().template cast<RankedTensorType>();
    if (!op.getAllowReorder()) {
      auto mod = op->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int threadsPerWarp =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
      if (srcTy.getEncoding() != triton::gpu::getDefaultBlockedEncoding(
                                     op.getContext(), srcTy.getShape(),
                                     numWarps, threadsPerWarp, numCTAs) ||
          resultTy.getEncoding() != triton::gpu::getDefaultBlockedEncoding(
                                        op.getContext(), resultTy.getShape(),
                                        numWarps, threadsPerWarp, numCTAs)) {
        return emitOptionalError(loc, "ReshapeOp lowering only supports the "
                                      "default block encoding right now.");
      }
    }

    auto typeConverter = getTypeConverter();
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    Value ret = packLLElements(loc, typeConverter, vals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct ExpandDimsOpConversion : public ConvertOpToLLVMPattern<ExpandDimsOp> {
  using OpAdaptor = typename ExpandDimsOp::Adaptor;
  explicit ExpandDimsOpConversion(LLVMTypeConverter &typeConverter,

                                  PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<ExpandDimsOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto resultTy = op.getType().template cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<SliceEncodingAttr>();
    if (!srcLayout) {
      return emitOptionalError(
          loc, "ExpandDimsOp only supports SliceEncodingAttr as its input");
    }

    auto resultLayout = resultTy.getEncoding();

    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    std::map<SmallVector<unsigned>, Value> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      offset.erase(offset.begin() + srcLayout.getDim());
      resultVals.push_back(srcValues.at(offset));
    }
    Value ret =
        packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct TransOpConversion : public ConvertOpToLLVMPattern<TransOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType().cast<RankedTensorType>();

    if (auto enc = resultTy.getEncoding().dyn_cast<SharedEncodingAttr>()) {
      auto llvmElemTy =
          getTypeConverter()->convertType(resultTy.getElementType());
      auto srcSmemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                        llvmElemTy, rewriter);
      auto dstSmemObj = SharedMemoryObject(
          srcSmemObj.base, srcSmemObj.baseElemType,
          /*strides=*/applyPermutation(srcSmemObj.strides, op.getOrder()),
          /*offsets=*/applyPermutation(srcSmemObj.offsets, op.getOrder()));
      auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
      rewriter.replaceOp(op, retVal);
      return success();
    } else if (auto enc =
                   resultTy.getEncoding().dyn_cast<BlockedEncodingAttr>()) {
      // If the dst encoding is blocked, then TransOp::inferReturnTypes
      // ensures that:
      //  - the src encoding is also blocked, and
      //  - the translation from src to dst is just a "renaming" of the
      //    registers, i.e. each thread has exactly the same values.
      // Thus the transpose op simply returns the same values it got.
      auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      Value ret = packLLElements(loc, this->getTypeConverter(), vals, rewriter,
                                 resultTy);
      rewriter.replaceOp(op, ret);
      return success();
    }

    return emitOptionalError(loc, "unsupported encoding for TransOp");
  }
};
} // namespace

void mlir::triton::populateViewOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<ReshapeOpConversion>(typeConverter, benefit);
  patterns.add<ExpandDimsOpConversion>(typeConverter, benefit);
  patterns.add<SplatOpConversion>(typeConverter, benefit);
  patterns.add<ArithConstantSplatOpConversion>(typeConverter, benefit);
  patterns.add<CatOpConversion>(typeConverter, benefit);
  patterns.add<InterleaveOpConversion>(typeConverter, benefit);
  patterns.add<TransOpConversion>(typeConverter, benefit);
}
