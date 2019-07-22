#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export CPATH=$CPATH:"${DIR}/compiler-rt/include"
