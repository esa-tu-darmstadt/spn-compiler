import random

from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, get_nodes_by_type, Sum, Product
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe

from spn_parser import load_spn_2


def load_spn(path: str):
  spn, var_2_index, _, _ = load_spn_2(path)
  return spn

def generate_spn_data(spn, count: int):
  pass

def get_spn_model(spn_path: str):
  spn = load_spn(spn_path)

  return

  inputs = generate_spn_data(spn, 100)
  outputs = log_likelihood(node=spn, data=inputs)

  return inputs, outputs

if __name__ == '__main__':
  print(f'loading ../../../examples/nips5.spn')
  load_spn('../../../examples/nips5.spn')

