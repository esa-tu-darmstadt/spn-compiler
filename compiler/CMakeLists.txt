find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(src/codegen/mlir/dialects/spn)

add_library(spnc SHARED src/Driver.cpp
        src/driver/toolchain/CPUToolchain.cpp
        src/driver/toolchain/MLIRToolchain.cpp
        src/driver/action/LLVMWriteBitcode.cpp
        src/driver/action/LLVMStaticCompiler.cpp
        src/driver/action/ClangKernelLinking.cpp
        src/driver/action/MLIRtoLLVMConversion.cpp
        src/graph-ir/GraphIRNode.cpp
        src/graph-ir/InputVar.cpp
        src/graph-ir/Histogram.cpp
        src/graph-ir/WeightedSum.cpp
        src/graph-ir/Sum.cpp
        src/graph-ir/Product.cpp
        src/json/Parser.cpp
        src/transform/BaseVisitor.cpp
        src/transform/IRTransformationPass.cpp
        src/util/DotVisitor.cpp
        src/util/GraphStatVisitor.cpp
        src/transform/BinaryTreeTransform.cpp
        src/codegen/llvm-ir/CPU/LLVMCPUCodegen.cpp
        src/codegen/llvm-ir/CPU/body/CodeGenScalarBody.cpp
        src/codegen/llvm-ir/CPU/loop/CodeGenSerialLoop.cpp
        src/codegen/mlir/dialects/spn/SPNDialect.cpp
        src/codegen/mlir/dialects/spn/SPNAttributes.cpp
        src/codegen/mlir/MLIRCodeGen.cpp
        src/codegen/mlir/MLIRBodyGen.cpp
        src/codegen/mlir/transform/pattern/CanonicalizationPatterns.cpp
        src/codegen/mlir/transform/pattern/SimplificationPatterns.cpp
        src/codegen/mlir/transform/passes/SPNOpSimplifier.cpp
        src/codegen/mlir/pipeline/MLIRPipeline.cpp
        src/codegen/mlir/lowering/patterns/LowerSPNtoStandardPatterns.cpp
        src/codegen/mlir/lowering/passes/SPNtoStandardLoweringPass.cpp
        src/codegen/mlir/lowering/patterns/LowerSPNtoLLVMPatterns.cpp
        src/codegen/mlir/lowering/passes/SPNToLLVMLoweringPass.cpp)

message(STATUS "MLIR Include:" ${MLIR_INCLUDE_DIRS})

target_include_directories(spnc
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common/include>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR})

set(mlir_libs
        MLIRAnalysis
        MLIRAffineOps
        MLIRAffineToStandard
        MLIRLoopsToGPU
        MLIRLinalgToLLVM
        MLIRLoopToStandard
        MLIREDSC
        MLIRExecutionEngine
        MLIRFxpMathOps
        MLIRGPU
        MLIRGPUtoNVVMTransforms
        MLIRGPUtoROCDLTransforms
        MLIRGPUtoSPIRVTransforms
        MLIRLinalgOps
        MLIRLLVMIR
        MLIRLoopOps
        MLIRNVVMIR
        MLIROpenMP
        MLIRParser
        MLIRPass
        MLIRQuantizerTransforms
        MLIRQuantOps
        MLIRROCDLIR
        MLIRSPIRV
        MLIRStandardToSPIRVTransforms
        MLIRSPIRVTestPasses
        MLIRSPIRVTransforms
        MLIRStandardOps
        MLIRStandardToLLVM
        MLIRTransforms
        MLIRTestDialect
        MLIRTestIR
        MLIRTestPass
        MLIRTestTransforms
        MLIRSupport
        MLIRVectorOps
        MLIRVectorToLLVM
        MLIRVectorToLoops
        )

llvm_map_components_to_libnames(llvm_libs bitwriter native)
target_link_libraries(spnc PRIVATE spnc-common ${mlir_libs} ${llvm_libs})

add_dependencies(spnc SPNOpsIncGen)
