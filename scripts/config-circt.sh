PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/capnproto/install;"

cmake -G Ninja .. \
    -DCMAKE_PREFIX_PATH=$PREFIX_PATH \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLLVM_ENABLE_RTTI=ON -DESI_COSIM=ON \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++
