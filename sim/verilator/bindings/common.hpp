#pragma once

#include <string>
#include <vector>

#include "Vspn_body.h"
#include "verilated.h"


class PySPNSim {
  std::unique_ptr<VerilatedContext> context;
  std::unique_ptr<Vspn_body> top;

  static uint8_t convertIndex(uint32_t input);
  static double convertProb(uint32_t prob);
public:
  void init(const std::vector<std::string>& args);
  void clock();
  void step();
  void setInput(const std::vector<uint32_t>& input);
  double getOutput() const;
  uint64_t getOutputRaw() const;
  void final();
};