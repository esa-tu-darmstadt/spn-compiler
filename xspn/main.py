import sys
from pathlib import Path

from xspn.spn_parser import load_spn
from xspn.serialization.binary.BinarySerialization import BinarySerializer, BinaryDeserializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability, ErrorKind


def print_usage():
    print("usage: ./main.py <path to .spn file>")

def bin_name(spn_name: str):
    return spn_name.split('.')[0] + '.bin'

if __name__ == '__main__':
    """
      1. A SPN is parsed and serialized into a binary file, a more efficient format.
      2. A kernel binary is compiled and returned to the calling python function.
      3. The queries can be executed by providing the kernel and the spn to the runtime (in binary format or whatever).

      The FPGA flow must be a little different:
      1. First, a SPN is converted into a binary file. That binary file is sent to the compiler that produces IP-XACT zip.
      2. That zip can be processes by Vivado via scripts. The resulting bitstream file is our kernel binary.
      3. The user then calls the SPN compiler with the bitstream. The compiler tries to load the bitstream via Tapasco onto the FPGA.
      4. Querying then happens through the usual interface as for CPUs and GPUs.

      Because bitstream synthesis is so slow the steps 1, 2 and 3, 4 are separated.
    """

    if len(sys.argv) <= 1:
        print_usage()
        exit()

    spn_path = Path(sys.argv[1])
    print(f'loading {spn_path}')

    spn, variables_to_index, index_to_min, index_to_max = load_spn(str(spn_path))

    model = SPNModel(spn, featureValueType='uint32')
    query = JointProbability(model)

    bin_path = Path(bin_name(spn_path.name))
    print(f'writing {bin_path}')
    BinarySerializer(str(bin_path)).serialize_to_file(query)
