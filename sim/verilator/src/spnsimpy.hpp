#pragma once

#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

extern "C" void initSim(int argc, const char **argv);
extern "C" void stepSim(void);
extern "C" void setInputSim(const std::vector<uint32_t> &input);
extern "C" double getOutputSim(void);
extern "C" void finalSim(void);

class NIPS5Sim {
public:
  NIPS5Sim() { initSim(0, nullptr); }

  NIPS5Sim(int argc, const char **argv) { initSim(argc, argv); }

  void step() { stepSim(); }

  void setInput(py::array &input) {
    assert(input.ndim() == 1);
    assert(input.shape()[0] == 5);
    // TODO: Assert correct dtype!

    const uint32_t *begin = reinterpret_cast<const uint32_t *>(input.data(0));
    const uint32_t *end = begin + 5;

    std::vector<uint32_t> inputVector(begin, end);
    setInputSim(inputVector);
  }

  double getOutput() { return getOutputSim(); }

  void final() { finalSim(); }
};

PYBIND11_MODULE(spnsimpy, m) {

  py::class_<NIPS5Sim>(m, "NIPS5Sim")
      //.def(py::init<>())
      .def("step", &NIPS5Sim::step);

  // py::class_<NIPS5Sim>(m, "NIPS5SimBase")
  //   .def(py::init<>(&createNIPS5Sim));

  // py::class_<NIPS5SimWrapper>(m, "NIPS5Sim")
  //   .def(py::init<>())
  //   .def("step", &NIPS5SimWrapper::step)
  //   .def("setInput", &NIPS5SimWrapper::setInput)
  //   .def("getOutput", &NIPS5SimWrapper::getOutput)
  //   .def("final", &NIPS5SimWrapper::final);
}