#pragma once

// TODO: Include top modules
#include "VFPAdd.h"
#include "VFPMult.h"
#include "verilated.h"

#include <iostream>

template <uint32_t ExpWidth, uint32_t ManWidth>
class TypeConverter {
public:
  static_assert(ExpWidth + ManWidth <= 64);

  static uint64_t toBits(double value) {
    float asFloat = float(value);
    return *reinterpret_cast<const uint32_t *>(&asFloat) & 0x7fffffff;
  }

  static double toDouble(uint64_t bits) {
    uint32_t asU32 = uint32_t(bits);
    return *reinterpret_cast<const float *>(&asU32);
  }
};

template <class Top, uint32_t ExpWidth, uint32_t ManWidth>
class USim {
  std::unique_ptr<VerilatedContext> context;
  std::unique_ptr<Top> top;

public:
  using TC = TypeConverter<ExpWidth, ManWidth>;

  void init(int argc, const char **argv) {
    context = std::make_unique<VerilatedContext>();
    context->commandArgs(argc, argv);
    top = std::make_unique<Top>(context.get());

    top->clock = 0;
    top->reset = 0;

    std::cout << "Initialized USim" << std::endl;
  }

  void setInput(double a, double b) {
    top->io_a = TC::toBits(a);
    top->io_b = TC::toBits(b);
  }

  bool step() {
    bool high = top->clock = !top->clock;
    top->eval();
    return high;
  }

  double getOutput() { return TC::toDouble(uint64_t(top->io_r)); }

  void final() { top->final(); }
};
