//
// Created by ls on 12/5/19.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "../../runtime/include/spnc-runtime.h"
#include "../../compiler/include/spnc.h"

namespace py = pybind11;

PYBIND11_MODULE(spncpy, m) {

  py::class_<Kernel>(m, "Kernel")
          .def(py::init<const std::string&, const std::string&>())
          .def("fileName", &Kernel::fileName)
          .def("kernelName", &Kernel::kernelName)
          .def("execute",
                  [](const Kernel& kernel, int num_elements, py::array_t<int>& inputs){
                      py::buffer_info input_buf = inputs.request();
                      if(input_buf.format != py::format_descriptor<int>::format()){
                        std::cerr << "ERROR: Expected an int array as input!" << std::endl;
                      }

                      auto result = py::array_t<double>(num_elements);
                      py::buffer_info output_buf = result.request();

                      void* input_ptr = (void*) input_buf.ptr;
                      double* output_ptr = (double*) output_buf.ptr;

                      spnc_rt::spn_runtime::instance().execute(kernel, num_elements, input_ptr, output_ptr);

                      return result;

                  });

  py::class_<spn_compiler>(m, "SPNCompiler")
          .def(py::init())
          .def("parseJSONString", &spn_compiler::parseJSONString);

  //m.def("execute", &execute);
}
