get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRSPN
        MLIROptLib
        MLIRSPNtoStandardTransforms
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslation
        MLIRSupport
        MLIRExecutionEngine
        )

add_llvm_executable(spnc-opt spnc-opt.cpp)

llvm_update_compile_flags(spnc-opt)
target_link_libraries(spnc-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(spnc-opt)