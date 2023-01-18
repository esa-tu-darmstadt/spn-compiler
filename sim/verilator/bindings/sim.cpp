#include "common.hpp"

#include <iostream>


void PySPNSim::init(const std::vector<std::string>& args) {
  std::cout << "init() called with the following arguments:\n";
  for (const auto& arg : args)
    std::cout << arg << "\n";
  
  std::vector<const char *> pointers;

  for (const std::string& s : args)
    pointers.push_back(s.c_str());
  
  context = std::make_unique<VerilatedContext>();
  context->commandArgs(pointers.size(), pointers.data());
  top = std::make_unique<Vspn_body>(context.get());

  std::cout << "Simulation initialized!\n";
}

void PySPNSim::clock() {
  top->clk = !top->clk;
}

void PySPNSim::step() {
  top->eval();
}

double PySPNSim::getOutput() const {
  return convertProb(top->out_prob);
}

uint64_t PySPNSim::getOutputRaw() const {
  return top->out_prob;
}

void PySPNSim::final() {
  top->final();
}