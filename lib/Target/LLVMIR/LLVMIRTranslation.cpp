#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.h"
#include "triton/tools/sys/getenv.hpp"
#include "llvm/IR/Constants.h"

namespace mlir {
namespace triton {

// Describes NVVM Metadata. It is used to record the nvvm related meta
// information from mlir module.
struct NVVMMetadata {
  int maxntidx{-1};
  bool is_kernel{};
  // Free to extend with other information.
};

// Add the nvvm related metadata to LLVM IR.
void amendLLVMFunc(llvm::Function *func, const NVVMMetadata &metadata) {
  auto *module = func->getParent();
  auto &ctx = func->getContext();

  if (metadata.maxntidx > 0) {
    auto i32_ty = llvm::IntegerType::get(ctx, 32);
    auto warps =
        llvm::ConstantInt::get(i32_ty, llvm::APInt(32, metadata.maxntidx));

    llvm::Metadata *md_args[] = {llvm::ValueAsMetadata::get(func),
                                 llvm::MDString::get(ctx, "maxntidx"),
                                 llvm::ValueAsMetadata::get(warps)};

    module->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(ctx, md_args));
  }

  if (metadata.is_kernel) {
    llvm::Metadata *md_args[] = {
        llvm::ValueAsMetadata::get(func), llvm::MDString::get(ctx, "kernel"),
        llvm::ValueAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
    module->getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(ctx, md_args));
  }
}

void extractNVVMMetadata(mlir::ModuleOp module,
                         llvm::DenseMap<llvm::StringRef, NVVMMetadata> *dic) {
  for (auto op : module.getOps<LLVM::LLVMFuncOp>()) {
    NVVMMetadata meta;

    bool hasMetadata{};

    // maxntid
    if (op->hasAttr(NVVMMetadataField::MaxNTid)) {
      auto attr = op->getAttr(NVVMMetadataField::MaxNTid);
      meta.maxntidx = attr.dyn_cast<IntegerAttr>().getInt();
      hasMetadata = true;
    }

    // kernel
    if (op->hasAttr(NVVMMetadataField::Kernel)) {
      meta.is_kernel = true;
      hasMetadata = true;
    }

    if (hasMetadata)
      dic->try_emplace(op.getNameAttr().strref(), std::move(meta));
  }
}

std::unique_ptr<llvm::Module>
translateLLVMToLLVMIR(llvm::LLVMContext *llvmContext, mlir::ModuleOp module) {
  auto context = module->getContext();
  DialectRegistry registry;
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  context->appendDialectRegistry(registry);

  llvm::DenseMap<llvm::StringRef, NVVMMetadata> nvvmMetadata;
  extractNVVMMetadata(module, &nvvmMetadata);

  auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return nullptr;
  }

  // Initialize LLVM targets.
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return nullptr;
  }

  for (auto &func : llvmModule->functions()) {
    auto it = nvvmMetadata.find(func.getName());
    if (it != nvvmMetadata.end())
      amendLLVMFunc(&func, it->second);
  }

  return llvmModule;
}

std::unique_ptr<llvm::Module>
translateTritonGPUToLLVMIR(llvm::LLVMContext *llvmContext,
                           mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());
  applyPassManagerCLOptions(pm);
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](mlir::Pass *pass, mlir::Operation *) {
        return ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
      },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

  pm.addPass(createConvertTritonGPUToLLVMPass());
  // Conanicalize to eliminate the remaining UnrealizedConversionCastOp
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass()); // Simplify the IR to improve readability.
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed";
    return nullptr;
  }

  auto llvmir = translateLLVMToLLVMIR(llvmContext, module);
  if (!llvmir) {
    llvm::errs() << "Translate to LLVM IR failed";
  }

  return llvmir;
}

} // namespace triton
} // namespace mlir
