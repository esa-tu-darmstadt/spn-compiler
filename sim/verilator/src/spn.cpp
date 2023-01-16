#include "spn.hpp"


uint8_t NIPS5Sim::convertIndex(uint32_t input) {
  return static_cast<uint8_t>(input);
}

double NIPS5Sim::convertProb(uint32_t prob) {
  return 0.0;
}

NIPS5Sim::NIPS5Sim(int argc, const char **argv) {
  context = std::make_unique<VerilatedContext>();
  context->commandArgs(argc, argv);
  top = std::make_unique<Vtop>(context.get());
  std::cout << "Initialized NIPS5 simulation" << std::endl;
}

void NIPS5Sim::step() {
  top->clk = !top->clk;
  top->eval();
}

void NIPS5Sim::setInput(const std::vector<uint32_t>& input) {
  assert(input.size() == 5);

  top->in_0 = convertIndex(input[0]);
  top->in_1 = convertIndex(input[1]);
  top->in_2 = convertIndex(input[2]);
  top->in_3 = convertIndex(input[3]);
  top->in_4 = convertIndex(input[4]);
}

double NIPS5Sim::getOutput() const {
  return convertProb(top->out_prob);
}

void NIPS5Sim::final() {
  top->final();
}