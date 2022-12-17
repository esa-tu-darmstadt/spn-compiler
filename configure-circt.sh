#cmake -DCMAKE_PREFIX_PATH="$BASE_DIR/llvm/build-llvm/lib/cmake/llvm;$BASE_DIR/llvm/build-llvm/lib/cmake/mlir;$BASE_DIR/pybind11/install/share/cmake/pybind11;$BASE_DIR/spdlog/install/lib64/cmake/spdlog;$BASE_DIR/capnproto/install"    -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_ASSERTIONS=ON    -DSPNC_BUILD_DOC=ON    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON    ..

#PREFIX_PATH="$BASE_DIR/llvm/build-llvm/lib/cmake/llvm;"
#PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/llvm/build-llvm/lib/cmake/mlir;"

PREFIX_PATH="$BASE_DIR/circt/circt/llvm/build/lib/cmake/llvm;"
PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/circt/circt/llvm/build/lib/cmake/mlir;"

PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/circt/circt/build/lib/cmake/circt;"

PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/pybind11/install/share/cmake/pybind11;"
PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/spdlog/install/lib/cmake/spdlog;"
PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/capnproto/install"

#echo $PREFIX_PATH

cmake -DCMAKE_PREFIX_PATH=$PREFIX_PATH\
    -DLLVM_ENABLE_LLD=ON\
    -DLLVM_ENABLE_ASSERTIONS=ON\
    -DSPNC_BUILD_DOC=ON\
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON\
    -DBUILD_SHARED_LIBS=ON\
    -DCMAKE_BUILD_TYPE=Debug\
    ..
