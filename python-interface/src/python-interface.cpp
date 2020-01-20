//
// Created by ls on 12/5/19.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../../runtime/include/spnc-runtime.h"

namespace py = pybind11;

PYBIND11_MODULE(spncpy, m) {

  py::class_<Kernel>(m, "Kernel")
          .def(py::init<const std::string&, const std::string&>())
          .def("fileName", &Kernel::fileName)
          .def("kernelName", &Kernel::kernelName)
          .def("execute",
                  [](const Kernel& kernel, int num_elements, py::array_t<double>& inputs){
                      py::buffer_info input_buf = inputs.request();

                      auto result = py::array_t<double>(input_buf.size);
                      py::buffer_info output_buf = result.request();

                      void* input_ptr = (void*) input_buf.ptr;
                      double* output_ptr = (double*) output_buf.ptr;

                      spnc_rt::spn_runtime::instance().execute(kernel, num_elements, input_ptr, output_ptr);

                      return result;

                  });

  //m.def("execute", &execute);
}
