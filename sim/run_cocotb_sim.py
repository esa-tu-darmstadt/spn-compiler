from pathlib import Path
import os
import subprocess
import tempfile

from xspn.structure.Model import SPNModel
from xspn.serialization.binary.BinarySerialization import BinarySerializer

from spnc.fpga import FPGACompiler


def sim_spn(spn_path):
    tmp_dir = Path(tempfile.mkdtemp())

    # load and serialize + compile network
    spn, var_2_index, index_2_min, index_2_max = load_spn_2(spn_path)

    compiler = FPGACompiler(computeInLogSpace=False)
    kernel = compiler.compile_ll(spn)

    # call cocotb
    env = {
        **os.environ.copy(),
        # path to the verilog sources
        'INCLUDE_DIR': str((tmp_dir / 'ipxact_core' / 'src').absolute())
        # TODO: insert path to the python testbench
    }

    cwd = ...

    subprocess.run('make pyspnsim', shell=True, cwd=cwd, env=env)


if __name__ == '__main__':
    spn_files = [p / 'structure.spn' for p in Path('resources/spns/').glob('*') if p.is_dir()]
    print(spn_files)


    Path('tmp') / ''
