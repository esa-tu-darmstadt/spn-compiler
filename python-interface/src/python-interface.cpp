//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
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
      .def(py::init<const std::string&, const std::string&, unsigned, unsigned, unsigned, unsigned>())
      .def("fileName", &Kernel::fileName)
      .def("kernelName", &Kernel::kernelName)
      .def("execute",
           [](const Kernel& kernel, int num_elements, py::array& inputs) {
             py::buffer_info input_buf = inputs.request();

             auto result = py::array_t<double>(num_elements);
             py::buffer_info output_buf = result.request();

             void* input_ptr = (void*) input_buf.ptr;
             auto* output_ptr = (double*) output_buf.ptr;

             spnc_rt::spn_runtime::instance().execute(kernel, num_elements, input_ptr, output_ptr);

             return result;

           });

  py::class_<spn_compiler>(m, "SPNCompiler")
      .def(py::init())
      .def("compileQuery", [](const spn_compiler& compiler, const std::string& inputFile, const options_t& options) {
        return spn_compiler::compileQuery(inputFile, options);
      });

}
