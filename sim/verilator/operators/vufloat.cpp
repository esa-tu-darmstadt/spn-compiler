#include "vufloat.hpp"


float randf() {
  return float(rand()) / RAND_MAX;
}

template <class Dut, uint32_t Delay, class Operator>
void test(int argc, const char **argv, Operator op) {
  Dut dut;
  dut.init(argc, argv);
  bool set = false;
  uint32_t delay = Delay;
  std::vector<double> exp;
  std::vector<double> got;

  for (uint32_t i = 0; i < 100; ++i) {
    if (set) {
      float a = randf() * 100, b = randf() * 100;
      //std::cout << "a = " << a << " b = " << b << " sum = " << a + b << "\n";
      exp.push_back(op(a, b));
      dut.setInput(a, b);

      if (delay == 0)
        got.push_back(dut.getOutput());
      else
        --delay;
    }

    set = !dut.step();    
  }

  dut.final();

  // compare
  std::cout << exp.size() - Delay << " " << got.size() << "\n";
  assert(exp.size() - Delay == got.size());

  for (size_t i = 0; i < got.size(); ++i) {
    std::cout << "exp = " << exp[i] << " got = " << got[i] << "\n";
  }
}

int main(int argc, const char **argv) {
  //std::cout << "Testing add:\n";
  //test<USim<VFPAdd, 8, 23>, 6>(argc, argv, [](double a, double b) { return a + b; });
  std::cout << "Testing mul:\n";
  test<USim<VFPMult, 8, 23>, 5>(argc, argv, [](double a, double b) { return a * b; });
  return 0;
}