#include <iostream>
#include <cstdint>

#include "Vcounter.h"
#include "verilated.h"


int main(int argc, const char **argv) {
  VerilatedContext *context = new VerilatedContext;
  context->commandArgs(argc, argv);

  std::cout << "Beginning simulation..." << std::endl;
  Vcounter *top = new Vcounter(context);

  for (uint32_t i = 0; i < 256; ++i) {
    top->clk = !top->clk;
    top->eval();

    std::cout << "Counter value: " << uint32_t(top->value) << "\n";
  }

  top->final();

  std::cout << "Done!" << std::endl;

  delete top;
  delete context;

  return 0;
}