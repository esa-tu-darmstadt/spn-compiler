#pragma once

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "spn.hpp"


namespace py = pybind11;


class NIPS5SimWrapper : public NIPS5Sim {
public:
  NIPS5SimWrapper(): NIPS5Sim(0, nullptr) {}

  void setInput(py::array& input) {
    assert(input.ndim() == 1);
    assert(input.shape()[0] == 5);
    // TODO: Assert correct dtype!
    
    const uint32_t *begin = reinterpret_cast<const uint32_t *>(input.data(0));
    const uint32_t *end = begin + 5;

    std::vector<uint32_t> inputVector(begin, end);
    NIPS5Sim::setInput(inputVector);
  }
};

NIPS5Sim createNIPS5Sim() {
  return NIPS5Sim(0, nullptr);
}

PYBIND11_MODULE(spnsimpy, m) {

  py::class_<NIPS5Sim>(m, "NIPS5SimBase")
    .def(py::init<>(&createNIPS5Sim));

  //py::class_<NIPS5SimWrapper>(m, "NIPS5Sim")
  //  .def(py::init<>())
  //  .def("step", &NIPS5SimWrapper::step)
  //  .def("setInput", &NIPS5SimWrapper::setInput)
  //  .def("getOutput", &NIPS5SimWrapper::getOutput)
  //  .def("final", &NIPS5SimWrapper::final);

}