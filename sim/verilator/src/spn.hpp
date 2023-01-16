#pragma once

#include "Vtop.h"
#include "verilated.h"

#include <iostream>
#include <vector>


// verilator simulation

class NIPS5Sim {
  std::unique_ptr<VerilatedContext> context;
  std::unique_ptr<Vtop> top;

  static uint8_t convertIndex(uint32_t input);
  static double convertProb(uint32_t prob);
public:

  NIPS5Sim(int argc, const char **argv);
  void step();
  void setInput(const std::vector<uint32_t>& input);
  double getOutput() const;
  void final();
};
