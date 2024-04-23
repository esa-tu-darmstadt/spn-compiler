#include <vector>
#include <cstdint>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


extern "C" void initSim(int argc, const char **argv);
extern "C" void stepSim(void);
extern "C" void setInputSim(const std::vector<uint32_t>& input);
extern "C" double getOutputSim(void);
extern "C" void finalSim(void);


//int main() {
//  initSim(0, nullptr);
//  return 0;
//}

void test() {
  std::cout << "test()\n";
  initSim(0, nullptr);
}

void _initSim() {
  initSim(0, nullptr);
}

PYBIND11_MODULE(mytest, m) {

  m.def("initSim", &_initSim);
  m.def("stepSim", &stepSim);
  m.def("setInputSim", [](py::array& input) {
    assert(input.ndim() == 1);
    assert(input.shape()[0] == 5);
    // TODO: Assert correct dtype!
    
    const uint32_t *begin = reinterpret_cast<const uint32_t *>(input.data(0));
    const uint32_t *end = begin + 5;

    std::vector<uint32_t> inputVector(begin, end);
    setInputSim(inputVector);
  });
  m.def("getOutputSim", &getOutputSim);
  m.def("finalSim", &finalSim);

}