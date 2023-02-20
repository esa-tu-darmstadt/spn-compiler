//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "../../runtime/include/spnc-runtime.h"
#include "../../compiler/include/spnc.h"

namespace py = pybind11;

PYBIND11_MODULE(spncpy, m) {

  py::class_<Kernel>(m, "Kernel")
      //.def(py::init<const std::string&, const std::string&, unsigned, unsigned, unsigned, unsigned,
      //              unsigned, unsigned, unsigned, const std::string&>())
      .def("fileName",
            [](const Kernel& kernel) {
              if (kernel.getKernelType() == KernelType::CLASSICAL_KERNEL)
                return kernel.getClassicalKernel().fileName;
              else
                return kernel.getFPGAKernel().fileName;
            })
      .def("kernelName",
            [](const Kernel& kernel) {
              if (kernel.getKernelType() == KernelType::FPGA_KERNEL)
                return kernel.getClassicalKernel().kernelName;
              else
                return kernel.getFPGAKernel().kernelName;
            })
      .def("execute",
           [](const Kernel& kernel, int num_elements, py::array& inputs) {
             if (kernel.getKernelType() == KernelType::CLASSICAL_KERNEL) {
              std::cout << "kernel type: " << kernel.getKernelType() << "\n";
              std::cout << "kernel index: " << kernel.kernel.index() << "\n";
              ClassicalKernel classical = kernel.getClassicalKernel();
              std::cout << "got classical\n";
             
              py::buffer_info input_buf = inputs.request();

              // Get a new array to hold the result values, using the data-type
              // and shape information attached to the kernel.
              auto dtype = py::dtype(classical.dtype);
              std::vector<unsigned> shape;
              shape.push_back(num_elements);
              if (classical.numResults > 1) {
                shape.push_back(classical.numResults);
              }
              auto result = py::array(dtype, shape);
              py::buffer_info result_buf = result.request(true);

              void* input_ptr = (void*) input_buf.ptr;
              void* output_ptr = (void*) result_buf.ptr;

              spnc_rt::spn_runtime::instance().execute(kernel, num_elements, input_ptr, output_ptr);

              return result;
             }

             throw std::runtime_error("not implemented");
           });

  py::class_<spn_compiler>(m, "SPNCompiler")
      .def(py::init())
      .def("compileQuery", [](const spn_compiler& compiler, const std::string& inputFile, const options_t& options) {
        return spn_compiler::compileQuery(inputFile, options);
      })
      .def("isTargetSupported", [](const std::string& target) {
        return spn_compiler::isTargetSupported(target);
      })
      .def("isFeatureAvailable", [](const std::string& feature) {
        return spn_compiler::isFeatureSupported(feature);
      })
      .def("getHostArchitecture", []() {
        return spn_compiler::getHostArchitecture();
      });

}
