﻿#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_passes_common(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
}

void init_triton_passes_ttir(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_0("add_combine", createCombineOpsPass);
  ADD_PASS_WRAPPER_0("add_reorder_broadcast", createReorderBroadcastPass);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     createRewriteTensorPointerPass);
  ADD_PASS_WRAPPER_4("add_convert_to_ttgpuir",
                     createConvertTritonToTritonGPUPass, int, int, int, int);
}

void init_triton_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton::gpu;
  ADD_PASS_WRAPPER_0("add_coalesce", createCoalescePass);
  ADD_PASS_WRAPPER_0("add_optimize_thread_locality",
                     createOptimizeThreadLocalityPass);
  ADD_PASS_WRAPPER_4("add_pipeline", createPipelinePass, int, int, int, int);
  ADD_PASS_WRAPPER_0("add_prefetch", createPrefetchPass);
  ADD_PASS_WRAPPER_1("add_accelerate_matmul", createAccelerateMatmulPass, int);
  ADD_PASS_WRAPPER_0("add_reorder_instructions", createReorderInstructionsPass);
  ADD_PASS_WRAPPER_0("add_optimize_dot_operands",
                     createOptimizeDotOperandsPass);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     createRemoveLayoutConversionsPass);
  ADD_PASS_WRAPPER_0("add_decompose_conversions",
                     createDecomposeConversionsPass);
}

void init_triton_passes_convert(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_scf_to_cf", createConvertSCFToCFPass);
  ADD_PASS_WRAPPER_0("add_index_to_llvm", createConvertIndexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_arith_to_llvm", createArithToLLVMConversionPass);
}

void init_triton_passes(py::module &&m) {
  init_triton_passes_common(m.def_submodule("common"));
  init_triton_passes_convert(m.def_submodule("convert"));
  init_triton_passes_ttir(m.def_submodule("ttir"));
  init_triton_passes_ttgpuir(m.def_submodule("ttgpuir"));
}
