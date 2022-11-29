# -*- Python -*-

import os
from pathlib import Path
import lit.formats
import lit.util

# Configuration file for the 'lit' test runner.

execute_external = (True)

# Process paths related to the custom opt tool
foo = os.path.normpath(config.spnc_opt_tool_bin_path)
config.spnc_opt_tool_bin_path = foo
spnc_opt_tool_bin_dir = os.path.abspath(os.path.dirname(config.spnc_opt_tool_bin_path))
ld_lib_path = os.path.pathsep.join((spnc_opt_tool_bin_dir, config.environment.get('LD_LIBRARY_PATH', '')))

# Set actual lit-config
# Note: Setting config.test_exec_root is needed to ensure that test-output is
# not written into the source directory (but into "build").
config.name = "spnc-mlir"
config.environment['LD_LIBRARY_PATH'] = ld_lib_path

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)
config.test_source_ex = os.path.normpath(os.path.join(config.test_source_root, "test-resources"))
config.excludes = [
    'analysis',

    'lowering/lospn-to-cpu',
    'lowering/standard-to-llvm',
    'transform'
]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.normpath(spnc_opt_tool_bin_dir)

config.test_format = lit.formats.ShTest(execute_external)
config.suffixes = ['.mlir']
config.substitutions.append(('%optcall', config.spnc_opt_tool_bin_path))
config.substitutions.append(('%drvcall', config.spnc_driver_bin_path))
config.substitutions.append(('%datadir', config.test_source_ex))
