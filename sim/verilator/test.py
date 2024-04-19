import mytest
import sys
import numpy as np
from spn_parser import load_spn_2
from spn.algorithms.Inference import likelihood


if __name__ == '__main__':
  if len(sys.argv) <= 1:
    print('Usage: test.py <spn path>')
    exit()

  spn_path = sys.argv[1]
  spn, _, _, _ = load_spn_2(spn_path)
  
  data = np.zeros((1, 5))
  print(f'prob = {likelihood(spn, data)}')

  mytest.initSim()

  for i in range(100):
    mytest.stepSim()
    print(mytest.getOutputSim())
