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
      .def(py::init<const std::string&, const std::string&, unsigned, unsigned, unsigned, unsigned,
                    unsigned, unsigned, unsigned, const std::string&>())
      .def("fileName", &Kernel::fileName)
      .def("kernelName", &Kernel::kernelName)
      .def("execute",
           [](const Kernel& kernel, int num_elements, py::array& inputs) {
             py::buffer_info input_buf = inputs.request();

             // Get a new array to hold the result values, using the data-type
             // and shape information attached to the kernel.
             auto dtype = py::dtype(kernel.dataType());
             std::vector<unsigned> shape;
             shape.push_back(num_elements);
             if (kernel.numResults() > 1) {
               shape.push_back(kernel.numResults());
             }
             auto result = py::array(dtype, shape);
             py::buffer_info result_buf = result.request(true);

             void* input_ptr = (void*) input_buf.ptr;
             void* output_ptr = (void*) result_buf.ptr;

             spnc_rt::spn_runtime::instance().execute(kernel, num_elements, input_ptr, output_ptr);

             return result;
           });

  py::class_<spn_compiler>(m, "SPNCompiler")
      .def(py::init())
      .def("compileQuery", [](const spn_compiler& compiler, const std::string& inputFile, const options_t& options) {
        return spn_compiler::compileQuery(inputFile, options);
      });

}
