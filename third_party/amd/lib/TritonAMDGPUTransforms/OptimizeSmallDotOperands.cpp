#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;

namespace {

class OptimizeFMADotPattern : public mlir::RewritePattern {

  triton::gpu::ConvertLayoutOp convertDotArg(mlir::PatternRewriter &rewriter,
                                             triton::DotOp dotOp, Value arg,
                                             Attribute newLayout) const {
    Location loc = dotOp.getLoc();
    rewriter.setInsertionPoint(dotOp);
    auto cvtTy = llvm::cast<RankedTensorType>(arg.getType());
    auto newArgTy = RankedTensorType::get(cvtTy.getShape(),
                                          cvtTy.getElementType(), newLayout);
    return rewriter.create<triton::gpu::ConvertLayoutOp>(loc, newArgTy, arg);
  }

  unsigned chooseSplitK(unsigned m, unsigned n, unsigned k, unsigned numWarps,
                        unsigned numThreads) const {
    // TODO experiment with this value
    return std::min(8u, k);
  }

  triton::gpu::BlockedEncodingAttr
  generateNewDotLayout(mlir::MLIRContext *ctx, unsigned m, unsigned n,
                       unsigned splitK, unsigned numWarps,
                       unsigned numThreads) const {
    assert(splitK > 0 && splitK <= numThreads);
    SmallVector<unsigned> order{2, 1, 0};
    triton::gpu::CTALayoutAttr ctaLayout =
        triton::gpu::CTALayoutAttr::get(ctx, {1, 1, 1}, {1, 1, 1}, {2, 1, 0});

    // TODO experiment with M/N first distribution of warps
    unsigned bWarps = 1;
    unsigned mWarps = std::min(m, numWarps);
    unsigned nWarps = numWarps / mWarps;
    SmallVector<unsigned> warpsPerCTA{bWarps, mWarps, nWarps};

    unsigned bThreads = splitK;
    unsigned nThreads = std::min(n / nWarps, numThreads / bThreads);
    unsigned mThreads = numThreads / bThreads / nThreads;
    SmallVector<unsigned> threadsPerWarp{bThreads, mThreads, nThreads};

    unsigned bElems = 1;
    unsigned mElems = std::max(1u, m / (mThreads * mWarps));
    unsigned nElems = std::max(1u, n / (nThreads * nWarps));
    SmallVector<unsigned> elemsPerThread{bElems, mElems, nElems};

    auto dotLayout = triton::gpu::BlockedEncodingAttr::get(
        ctx, elemsPerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);
    return dotLayout;
  }

  std::optional<RankedTensorType> splitTypeDim(RankedTensorType ty, int axis,
                                               int splitSize,
                                               Location loc) const {
    auto origShape = ty.getShape();
    SmallVector<int64_t> shape{origShape};
    assert(shape[axis] % splitSize == 0);
    shape.insert(shape.begin() + axis, splitSize);
    shape[axis + 1] /= splitSize;
    Attribute splitLayout;
    auto origEncoding = ty.getEncoding();
    auto layoutHelper = llvm::cast<triton::DialectInferLayoutInterface>(
        &origEncoding.getDialect());
    auto layoutInferResult = layoutHelper->inferReshapeOpNoReorderEncoding(
        origShape, origEncoding, shape, splitLayout, loc);
    if (layoutInferResult.failed())
      return std::nullopt;
    auto newType =
        RankedTensorType::get(shape, ty.getElementType(), splitLayout);
    return newType;
  }

  triton::gpu::BlockedEncodingAttr
  transposeLayout(triton::gpu::BlockedEncodingAttr layout,
                  ArrayRef<int> permOrder) const {
    auto ctx = layout.getContext();
    triton::gpu::CTALayoutAttr ctaLayout =
        triton::gpu::CTALayoutAttr::get(ctx, {1, 1, 1}, {1, 1, 1}, {2, 1, 0});
    // TODO adjust this if better performance is possible
    SmallVector<unsigned> order{2, 1, 0};
    SmallVector<unsigned> sizePerThread;
    SmallVector<unsigned> threadsPerWarp;
    SmallVector<unsigned> warpsPerCTA;
    for (auto idx : permOrder) {
      sizePerThread.push_back(layout.getSizePerThread()[idx]);
      threadsPerWarp.push_back(layout.getThreadsPerWarp()[idx]);
      warpsPerCTA.push_back(layout.getWarpsPerCTA()[idx]);
    }
    return triton::gpu::BlockedEncodingAttr::get(
        ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);
  }

  triton::gpu::BlockedEncodingAttr
  convertDotOpBToBlockedLayout(triton::gpu::DotOperandEncodingAttr layout,
                               int kSize) const {
    auto parentLayout =
        cast<triton::gpu::BlockedEncodingAttr>(layout.getParent());
    auto ctx = layout.getContext();
    SmallVector<unsigned> sizePerThread(parentLayout.getSizePerThread());
    sizePerThread[1] = kSize;
    SmallVector<unsigned> threadsPerWarp(parentLayout.getThreadsPerWarp());
    auto numWarps = triton::gpu::getNumWarpsPerCTA(parentLayout);
    SmallVector<unsigned> warpsPerCTA{1, 1, numWarps};
    SmallVector<unsigned> order(parentLayout.getOrder());
    auto ctaLayout = parentLayout.getCTALayout();
    return triton::gpu::BlockedEncodingAttr::get(
        ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);
  }

  triton::gpu::BlockedEncodingAttr
  mergeSlowDimLayoutToFaster(triton::gpu::BlockedEncodingAttr layout,
                             int slowDim) const {
    int rank = layout.getOrder().size();
    assert(slowDim >= 0 && slowDim < rank - 1);
    assert(slowDim != layout.getOrder()[0]);
    auto ctx = layout.getContext();
    SmallVector<unsigned> order;
    int fasterDim = -1;
    for (int i = 0; i < rank; ++i) {
      int dimIdx = layout.getOrder()[i];
      if (dimIdx > slowDim)
        order.push_back(dimIdx - 1);
      if (dimIdx < slowDim)
        order.push_back(dimIdx);
      if (dimIdx == slowDim)
        fasterDim = layout.getOrder()[i - 1];
    }
    SmallVector<unsigned> sizePerThread(layout.getSizePerThread());
    sizePerThread[fasterDim] *= sizePerThread[slowDim];
    sizePerThread.erase(sizePerThread.begin() + slowDim);

    SmallVector<unsigned> threadsPerWarp(layout.getThreadsPerWarp());
    threadsPerWarp[fasterDim] *= threadsPerWarp[slowDim];
    threadsPerWarp.erase(threadsPerWarp.begin() + slowDim);

    SmallVector<unsigned> warpsPerCTA(layout.getWarpsPerCTA());
    warpsPerCTA[fasterDim] *= warpsPerCTA[slowDim];
    warpsPerCTA.erase(warpsPerCTA.begin() + slowDim);
    assert(rank == 3);
    auto ctaLayout =
        triton::gpu::CTALayoutAttr::get(ctx, {1, 1}, {1, 1}, {1, 0});
    return triton::gpu::BlockedEncodingAttr::get(
        ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);
  }

  mlir::LogicalResult optimizeLargeBOp(mlir::PatternRewriter &rewriter,
                                       unsigned m, unsigned n, unsigned k,
                                       triton::DotOp dotOp,
                                       triton::gpu::ConvertLayoutOp aCvt,
                                       triton::gpu::ConvertLayoutOp bCvt,
                                       triton::LoadOp bLoadOp) const {
    auto loadTy = llvm::cast<RankedTensorType>(bLoadOp.getResult().getType());
    auto oldBLoadLayout = loadTy.getEncoding();
    auto ctx = dotOp.getContext();
    auto cOrigType = dotOp.getC().getType();
    auto cOrigLayout = llvm::dyn_cast<triton::gpu::BlockedEncodingAttr>(
        cOrigType.getEncoding());
    auto dotLoc = dotOp.getLoc();
    assert(cOrigLayout &&
           "non-blocked layouts are filtered outside this pattern");
    unsigned numWarps = triton::gpu::getNumWarpsPerCTA(cOrigLayout);
    unsigned numThreads = product(triton::gpu::getThreadsPerWarp(cOrigLayout));
    auto splitK = chooseSplitK(m, n, k, numWarps, numThreads);

    // create new dot output and argument layouts
    auto cBatchedOpLayout =
        generateNewDotLayout(ctx, m, n, splitK, numWarps, numThreads);
    auto elTy = bCvt.getType().getElementType();
    auto aBatchedOpLayout = triton::gpu::DotOperandEncodingAttr::get(
        ctx, 0, cBatchedOpLayout, elTy);
    auto bBatchedOpLayout = triton::gpu::DotOperandEncodingAttr::get(
        ctx, 1, cBatchedOpLayout, elTy);

    // create layouts and values related to A operand
    //
    // A operand data flow before transformation:
    //   aOrig (NxK aOrigLayout)   -layout_convert->
    //         (NxK dot op layout)
    // After transformation:
    //   aOrig      (MxK aOrigLayout)         -reshape->
    //   aExtended  (MxSxK' aExtendedLayout)  -trans->
    //   aBatched   (SxMxK' aBatchedLayout)   -layout_convert->
    //   aBatchedOp (SxMxK' aBatchedOpLayout)
    // "S" in shapes represents splitK value
    auto aOrig = aCvt.getSrc();
    auto aCvtLoc = aCvt.getLoc();
    auto aOrigLayout = dyn_cast<triton::gpu::BlockedEncodingAttr>(
        aOrig.getType().getEncoding());
    if (!aOrigLayout)
      return failure();
    auto aOrigType = aOrig.getType();
    auto optAExtendedType = splitTypeDim(aOrigType, 1, splitK, aCvtLoc);
    if (!optAExtendedType.has_value())
      return failure();
    auto aExtendedType = optAExtendedType.value();
    auto aExtendedLayout = llvm::cast<triton::gpu::BlockedEncodingAttr>(
        aExtendedType.getEncoding());
    rewriter.setInsertionPoint(aCvt);
    assert(!triton::gpu::isExpensiveView(aOrig.getType(), aExtendedType));
    auto aExtended =
        rewriter.create<triton::ReshapeOp>(aCvtLoc, aExtendedType, aOrig, true);
    auto aBatchedLayout = transposeLayout(aExtendedLayout, {1, 0, 2});
    auto aBatchedType =
        RankedTensorType::get({splitK, m, k / splitK}, elTy, aBatchedLayout);
    auto aBatched = rewriter.create<triton::TransOp>(
        aCvtLoc, aExtended, ArrayRef<int32_t>({1, 0, 2}));
    auto aBatchedOpType =
        RankedTensorType::get({splitK, m, k / splitK}, elTy, aBatchedOpLayout);
    auto aBatchedOp = rewriter.create<triton::gpu::ConvertLayoutOp>(
        dotLoc, aBatchedOpType, aBatched);

    // create layouts and values related to B operand
    //
    // B operand data flow before transformation:
    //   bOrigAddr, bOrigMask (KxN bOrigLoadLayout) -global load->
    //   bOrigLoad            (KxN bOrigLoadLayout) -layout_convert->
    //                        (KxN dot op layout)
    // After transformation:
    //   bOrigAddr, bOrigMask (KxN bOrigLoadLayout)    -layout_convert->
    //   bAddr, bMask         (KxN bLoadLayout)        -global load->
    //   bLoad                (KxN bLoadLayout)        -reshape->
    //   bBatched             (SxK'xN bBatchedLayout)  -layout_convert->
    //   bBatchedOp           (SxK'xN bBatchedOpLayout)
    auto bOrigAddr = bLoadOp.getPtr();
    auto bOrigMask = bLoadOp.getMask();
    auto bLoadLoc = bLoadOp.getLoc();

    auto bBatchedLayout =
        convertDotOpBToBlockedLayout(bBatchedOpLayout, k / splitK);
    auto bLoadLayout = mergeSlowDimLayoutToFaster(bBatchedLayout, 0);
    auto bLoadElTy = llvm::cast<RankedTensorType>(bLoadOp.getPtr().getType())
                         .getElementType();
    rewriter.setInsertionPoint(bLoadOp);
    auto bAddrType = RankedTensorType::get({k, n}, bLoadElTy, bLoadLayout);
    auto bAddr = rewriter.create<triton::gpu::ConvertLayoutOp>(
        bLoadLoc, bAddrType, bOrigAddr);
    Value bMask;
    if (bOrigMask) {
      auto bOrigMaskElTy =
          cast<RankedTensorType>(bOrigMask.getType()).getElementType();
      auto bMaskType =
          RankedTensorType::get({k, n}, bOrigMaskElTy, bLoadLayout);
      bMask = rewriter.create<triton::gpu::ConvertLayoutOp>(bLoadLoc, bMaskType,
                                                            bOrigMask);
    }
    auto bLoad = rewriter.create<triton::LoadOp>(
        bLoadLoc, bAddr, bMask, bLoadOp.getCache(), bLoadOp.getEvict(),
        bLoadOp.getIsVolatile());
    auto bBatchedType =
        RankedTensorType::get({splitK, k / splitK, n}, elTy, bBatchedLayout);
    assert(!triton::gpu::isExpensiveView(bLoad.getType(), bBatchedType));
    auto bBatched =
        rewriter.create<triton::ReshapeOp>(bLoadLoc, bBatchedType, bLoad, true);
    auto bBatchedOpType =
        RankedTensorType::get({splitK, k / splitK, n}, elTy, bBatchedOpLayout);
    auto bBatchedOp = rewriter.create<triton::gpu::ConvertLayoutOp>(
        dotLoc, bBatchedOpType, bBatched);

    // create layouts and values related to C operand
    //
    // C operand data flow before transformation:
    //   cOrig (MxN cOrigLayout)
    // After transformation:
    //   cOrig      (MxN cOrigLayout)   -reshape->
    //   cShaped    (1xMxN cOrigLayout) -broadcast->
    //   cBatched   (SxMxN cOrigLayout) -layout_convert->
    //   cBatchedOp (SxMxN cLayout)
    auto cOrig = dotOp.getC();
    auto cElTy = dotOp.getC().getType().getElementType();
    auto optCShapedType = splitTypeDim(cOrigType, 0, 1, cOrig.getLoc());
    if (!optCShapedType.has_value())
      return failure();
    auto cShapedType = optCShapedType.value();
    auto cShapedLayout = cShapedType.getEncoding();
    rewriter.setInsertionPoint(dotOp);
    assert(!triton::gpu::isExpensiveView(cOrig.getType(), cShapedType));
    auto cShaped =
        rewriter.create<triton::ReshapeOp>(dotLoc, cShapedType, cOrig, false);
    auto cBatchedType =
        RankedTensorType::get({splitK, m, n}, cElTy, cShapedLayout);
    auto cBatched =
        rewriter.create<triton::BroadcastOp>(dotLoc, cBatchedType, cShaped);
    auto cBatchedOpType =
        RankedTensorType::get({splitK, m, n}, cElTy, cBatchedOpLayout);
    auto cBatchedOp = rewriter.create<triton::gpu::ConvertLayoutOp>(
        dotLoc, cBatchedOpType, cBatched);

    // Create new dot and reduction
    rewriter.setInsertionPoint(dotOp);

    auto newDot = rewriter.create<triton::DotOp>(
        dotLoc, aBatchedOp, bBatchedOp, cBatchedOp, dotOp.getInputPrecision(),
        dotOp.getMaxNumImpreciseAcc());

    auto reduceOp = rewriter.create<triton::ReduceOp>(
        dotLoc, ArrayRef<Value>{newDot.getD()}, 0);
    auto reduceConvert =
        rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
            dotOp, cOrigType, reduceOp.getResult());

    auto reduceBody = rewriter.createBlock(&reduceOp.getRegion(), {},
                                           {cElTy, cElTy}, {dotLoc, dotLoc});
    auto blockArgs = reduceBody->getArguments();
    rewriter.setInsertionPoint(reduceBody, reduceBody->begin());
    Value reduceOperation;
    if (elTy.isInteger())
      reduceOperation =
          rewriter.create<arith::AddIOp>(dotLoc, blockArgs[0], blockArgs[1]);
    else
      reduceOperation =
          rewriter.create<arith::AddFOp>(dotLoc, blockArgs[0], blockArgs[1]);
    rewriter.create<triton::ReduceReturnOp>(dotLoc, reduceOperation);
    return success();
  }

public:
  explicit OptimizeFMADotPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<triton::DotOp>(op);
    auto dEncoding = dotOp.getD().getType().getEncoding();
    if (!llvm::dyn_cast<triton::gpu::BlockedEncodingAttr>(dEncoding))
      return failure();
    auto aCvt = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        dotOp.getA().getDefiningOp());
    auto bCvt = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        dotOp.getB().getDefiningOp());
    auto cOp = dotOp.getC();
    if (!aCvt || !bCvt)
      return mlir::failure();
    auto cShape = cOp.getType().getShape();
    auto aShape = aCvt.getType().getShape();
    if (cShape.size() != 2)
      return mlir::failure();
    unsigned m = cShape[0];
    unsigned n = cShape[1];
    unsigned k = aShape[1];
    if (m <= 8 && n >= 8) {
      auto loadOp =
          llvm::dyn_cast_or_null<triton::LoadOp>(bCvt.getSrc().getDefiningOp());
      if (!loadOp)
        return mlir::failure();
      return optimizeLargeBOp(rewriter, m, n, k, dotOp, aCvt, bCvt, loadOp);
    }
    return mlir::failure();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUOptimizeSmallDotOperandsPass
    : public TritonAMDGPUOptimizeSmallDotOperandsBase<
          TritonAMDGPUOptimizeSmallDotOperandsPass> {

public:
  TritonAMDGPUOptimizeSmallDotOperandsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    mlir::RewritePatternSet patterns(context);

    patterns.add<OptimizeFMADotPattern>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUOptimizeSmallDotOperandsPass() {
  return std::make_unique<TritonAMDGPUOptimizeSmallDotOperandsPass>();
}
