#include "spn.hpp"


static NIPS5Sim sim;

uint8_t NIPS5Sim::convertIndex(uint32_t input) {
  return static_cast<uint8_t>(input);
}

double NIPS5Sim::convertProb(uint32_t prob) {
  uint64_t p = prob;
  return *reinterpret_cast<const double *>(&p);
}

NIPS5Sim::NIPS5Sim(int argc, const char **argv) {
  context = std::make_unique<VerilatedContext>();
  context->commandArgs(argc, argv);
  top = std::make_unique<Vtop>(context.get());

  setInput({0, 0, 0, 0, 0});
  top->clk = 0;
  top->rst = 0;

  std::cout << "Initialized NIPS5 simulation -  all signals set to 0" << std::endl;
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

void initSim(int argc, const char **argv) {
  sim = NIPS5Sim(argc, argv);
}

void stepSim(void) {
  sim.step();
}

void setInputSim(const std::vector<uint32_t>& input) {
  sim.setInput(input);
}

double getOutputSim(void) {
  return sim.getOutput();
}

void finalSim(void) {
  sim.final();
}
