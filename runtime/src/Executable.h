//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_EXECUTABLE_H
#define SPNC_EXECUTABLE_H

#include <cstdlib>
#include <Kernel.h>

using namespace spnc;

namespace spnc_rt {

  typedef void (* kernel_function)(void* input_ptr,
                                   void* aligned_input_ptr,
                                   int64_t input_offset,
                                   int64_t input_size_dim1,
                                   int64_t input_size_dim2,
                                   int64_t input_stride_dim1,
                                   int64_t input_stride_dim2,
                                   void* output_ptr,
                                   void* output_aligned_ptr,
                                   int64_t output_offset,
                                   int64_t output_size_dim1,
                                   int64_t output_size_dim2,
                                   int64_t output_stride_dim1,
                                   int64_t output_stride_dim2);

  ///
  /// Manages a Kernel by loading it from the shared object using libelf.
  class Executable {

  public:

    /// Constructor.
    /// \param kernel Kernel to load and eventually execute.
    explicit Executable(const Kernel& kernel);

    Executable(const Executable&) = delete;

    Executable& operator=(const Executable&) = delete;

    /// Move constructor.
    /// \param other Move source.
    Executable(Executable&& other) noexcept;

    /// Move assignment operator.
    /// \param other Move source.
    /// \return Reference to move target.
    Executable& operator=(Executable&& other) noexcept;

    ~Executable();

    /// Execute the Kernel.
    /// \param num_elements Number of queries in the batch.
    /// \param inputs Input SPN evidence.
    /// \param outputs SPN output probabilities.
    void execute(size_t num_elements, void* inputs, void* outputs);

  private:
    Kernel kernel;

    void* handle;

    kernel_function kernel_func;

    void initialize();

    void executeSingle(size_t num_samples, void* inputs, void* outputs);

    void executeBatch(size_t num_samples, void* inputs, void* outputs);

    void executeGPU(size_t num_samples, void* inputs, void* outputs);
  };

  kernel_function tapasco_get_kernel_func(const Kernel& kernel);
}

#endif //SPNC_EXECUTABLE_H
