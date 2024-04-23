#pragma once

#include <cstdint>
#include <vector>

class TypeConverter {
public:
};

class Sim {
public:
  virtual void init(int argc, const char **argv) = 0;
  virtual void step() = 0;
  virtual void setInput(const std::vector<uint8_t> &bytes) = 0;
  virtual void getOutput() = 0;
  virtual void final() = 0;
};