//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../compiler/include/spnc.h"
#include "../../runtime/include/spnc-runtime.h"
#include "Kernel.h"

#ifdef SPNC_IPU_SUPPORT
#include "../../runtime/ipu/IPURuntime.h"
#endif

namespace py = pybind11;

PYBIND11_MODULE(spncpy, m) {

  py::class_<Kernel>(m, "Kernel")
      .def("execute",
           [](const Kernel &kernel, int num_elements, py::array &inputs) {
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

             void *input_ptr = (void *)input_buf.ptr;
             void *output_ptr = (void *)result_buf.ptr;

             spnc_rt::spn_runtime::instance().execute(kernel, num_elements,
                                                      input_ptr, output_ptr);

             return result;
           });

  py::class_<SharedObjectKernel, Kernel>(m, "SharedObjectKernel")
      .def(py::init<const std::string &, const std::string &, unsigned,
                    unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                    const std::string &>())
      .def("fileName", &SharedObjectKernel::fileName)
      .def("kernelName", &SharedObjectKernel::kernelName);

  py::class_<spn_compiler>(m, "SPNCompiler")
      .def(py::init())
      .def("compileQuery",
           [](const spn_compiler &compiler, const std::string &inputFile,
              const options_t &options) {
             return spn_compiler::compileQuery(inputFile, options);
           })
      .def("isTargetSupported",
           [](const std::string &target) {
             return spn_compiler::isTargetSupported(target);
           })
      .def("isFeatureAvailable",
           [](const std::string &feature) {
             return spn_compiler::isFeatureSupported(feature);
           })
      .def("getHostArchitecture",
           []() { return spn_compiler::getHostArchitecture(); });

#ifdef SPNC_IPU_SUPPORT
  {
    py::module ipu = m.def_submodule("ipu");

    py::enum_<spnc::IPUTarget>(ipu, "IPUTarget")
        .value("MODEL", spnc::IPUTarget::Model)
        .value("IPU1", spnc::IPUTarget::IPU1)
        .value("IPU2", spnc::IPUTarget::IPU2)
        .value("IPU21", spnc::IPUTarget::IPU21)
        .export_values();

    py::class_<spnc_rt::ipu::IPURuntime>(ipu, "IPURuntime")
        .def(py::init<>())
        .def("unload", &spnc_rt::ipu::IPURuntime::unload,
             "Detach from the current IPU target")
        .def("load", &spnc_rt::ipu::IPURuntime::load, py::arg("kernel"),
             "Load a kernel into the IPU")
        .def(
            "execute",
            [](spnc_rt::ipu::IPURuntime &self, int num_elements,
               py::array &inputs) {
              spnc::IPUKernel &kernel = self.getLoadedKernel();

              // Get a new array to hold the result values, using the data-type
              // and shape information attached to the kernel.
              auto dtype = py::dtype(kernel.dataType());
              std::vector<unsigned> shape;
              shape.push_back(num_elements);
              if (kernel.numResults() > 1) {
                shape.push_back(kernel.numResults());
              }
              auto result = py::array(dtype, shape);

              py::buffer_info inputs_info = inputs.request();
              py::buffer_info outputs_info = result.request(true);

              self.execute(num_elements, inputs_info.ptr, outputs_info.ptr);

              return result;
            },
            py::arg("num_elements"), py::arg("inputs").noconvert(),
            "Execute the loaded kernel with given inputs and outputs");
  }
#endif
}
