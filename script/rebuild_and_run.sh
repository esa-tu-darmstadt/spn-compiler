#!/bin/bash
set -e
ninja
./execute/driver $1 $3
dot -Tpng spn.dot -o spn.png
#/Users/johannesschulte/Desktop/Uni/MT/llvm-project/build/bin/clang++ -ffast-math -mfma -march=skylake -I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1 -L/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk -ffp-contract=fast -O3 ../debLLVMBuild/out.bc ../execute/resources/main.cpp
/usr/local/Cellar/llvm/9.0.0_1/bin/clang++ -ffast-math -mfma -march=skylake -ffp-contract=fast -O3 ../debLLVMBuild/out.bc ../execute/resources/main.cpp
./a.out $2inputdata.txt $2outputdata.txt a
