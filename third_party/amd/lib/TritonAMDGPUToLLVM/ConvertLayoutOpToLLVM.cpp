#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace SharedToDotOperandMFMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandMFMA

namespace SharedToDotOperandWMMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandWMMA

namespace {
struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (dstLayout.isa<DotOperandEncodingAttr>() &&
        dstLayout.cast<DotOperandEncodingAttr>()
            .getParent()
            .isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>()) {
      return lowerSharedToDotOperand(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  // shared -> dot_operand if the result layout is mfma
  Value lowerSharedToDotOperandMMA(
      triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
      const LLVMTypeConverter *typeConverter,
      ConversionPatternRewriter &rewriter,
      const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto llvmElemTy = typeConverter->convertType(
        src.getType().cast<MemDescType>().getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    Value res;
    auto dopOpParent = dotOperandLayout.getParent();
    if (!isOuter &&
        dopOpParent.isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>()) {
      auto sharedToDotConvert = dopOpParent.isa<AMDMfmaEncodingAttr>()
                                    ? SharedToDotOperandMFMA::convertLayout
                                    : SharedToDotOperandWMMA::convertLayout;
      res = sharedToDotConvert(dotOperandLayout.getOpIdx(), rewriter, loc, src,
                               dotOperandLayout, smemObj, typeConverter,
                               tid_val());
    } else {
      assert(false && "unsupported layout found");
    }
    return res;
  }

  // shared -> matrix_core_dot_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::LocalLoadOp op,
                          triton::gpu::LocalLoadOpAdaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto dstTensorTy = dst.getType().cast<RankedTensorType>();
    auto srcTensorTy = src.getType().cast<MemDescType>();
    auto dotOperandLayout =
        dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto sharedLayout = srcTensorTy.getEncoding().cast<SharedEncodingAttr>();

    bool isOuter{};
    int K{};
    if (dotOperandLayout.getOpIdx() == 0) // $a
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[1]];
    isOuter = K == 1;
    Value res = lowerSharedToDotOperandMMA(op, adaptor, typeConverter, rewriter,
                                           dotOperandLayout, isOuter);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::ConvertLayoutOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (srcLayout.isa<AMDMfmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMfmaToDotOperand(op, adaptor, rewriter);
    }
    return failure();
  }

private:
  LogicalResult
  lowerMfmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    if (isMfmaToDotShortcut(srcTy, dstTy)) {
      // vecSize is an number of sequential elements stored by one thread
      // - For MFMA encoding (encoding of the result tensor of dot
      // operation) it is 4
      // - For MFMA operand encoding it is
      // dotOperandEncoding::kWidth,
      //   which is 4 in certain cases (e.g. fp16 and bfloat16 dtypes with kpack
      //   = 1)
      //
      // For cases where these two values are equal MFMA and MFMA operand
      // layouts are the same.
      auto dstLayout = dstTy.getEncoding();
      auto dotOperandLayout = dstLayout.cast<DotOperandEncodingAttr>();
      assert(dotOperandLayout.getOpIdx() == 0 &&
             dotOperandLayout.getKWidth() == 4);
      // get source values
      auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      Value view =
          packLLElements(loc, getTypeConverter(), vals, rewriter, dstTy);
      rewriter.replaceOp(op, view);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace mlir::triton::AMD {
void populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, benefit);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
} // namespace mlir::triton::AMD
