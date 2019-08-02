#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export CPATH=$CPATH:"${DIR}/compiler-rt/include"
export LIBRARY_PATH=$LIBRARY_PATH:"${DIR}/compiler-rt/include/posit/lib"
