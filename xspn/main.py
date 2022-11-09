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
