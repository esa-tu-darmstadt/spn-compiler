#cmake -DCMAKE_PREFIX_PATH="$BASE_DIR/llvm/build-llvm/lib/cmake/llvm;$BASE_DIR/llvm/build-llvm/lib/cmake/mlir;$BASE_DIR/pybind11/install/share/cmake/pybind11;$BASE_DIR/spdlog/install/lib64/cmake/spdlog;$BASE_DIR/capnproto/install"    -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_ASSERTIONS=ON    -DSPNC_BUILD_DOC=ON    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON    ..

#PREFIX_PATH="$BASE_DIR/llvm/build-llvm/lib/cmake/llvm;"
#PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/llvm/build-llvm/lib/cmake/mlir;"

PREFIX_PATH="$BASE_DIR/circt/llvm/build/lib/cmake/llvm;"
PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/circt/llvm/build/lib/cmake/mlir;"

PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/circt/build/lib/cmake/circt;"

PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/pybind11/install/share/cmake/pybind11;"
PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/spdlog/install/lib64/cmake/spdlog;"
PREFIX_PATH=$PREFIX_PATH"$BASE_DIR/capnproto/install;"

#echo $PREFIX_PATH

# https://bugzilla.redhat.com/show_bug.cgi?id=2140764
cmake -DCMAKE_PREFIX_PATH=$PREFIX_PATH\
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"\
    -DLLVM_ENABLE_LLD=ON\
    -DCMAKE_C_COMPILER=/usr/bin/clang\
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++\
    -DCMAKE_CXX_FLAGS="-fuse-ld=lld"\
    -DLLVM_ENABLE_ASSERTIONS=ON\
    -DSPNC_BUILD_DOC=ON\
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON\
    -DBUILD_SHARED_LIBS=ON\
    -DCMAKE_BUILD_TYPE=Debug\
    ..
