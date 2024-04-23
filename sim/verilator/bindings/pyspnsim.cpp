#include "pyspnsim.hpp"


namespace py = pybind11;


PYBIND11_MODULE(pyspnsim, m) {
  py::class_<PySPNSim>(m, "PySPNSim")
    .def(py::init<>())
    .def("init", [](PySPNSim& self, py::list args) {
      std::vector<std::string> sargs;

      for (const auto& item : args)
        sargs.push_back(
          item.attr("__str__")().cast<std::string>()
        );

      self.init(sargs);
    })
    .def("clock", &PySPNSim::clock)
    .def("step", &PySPNSim::step)
    .def("setInput", [](PySPNSim& self, py::array input) {
      assert(input.ndim() == 1);
      //assert(input.shape()[0] == 1);
      // TODO: Assert correct dtype!

      const uint32_t *begin = reinterpret_cast<const uint32_t *>(input.data(0));
      const uint32_t *end = begin + input.shape()[0];

      std::vector<uint32_t> inputVector(begin, end);
      self.setInput(inputVector);
    })
    .def("getOutput", &PySPNSim::getOutput)
    .def("getOutputRaw", &PySPNSim::getOutputRaw)
    .def("final", &PySPNSim::final);
}
