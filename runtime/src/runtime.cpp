//
// Created by lukas on 22.11.19.
//

#include "../include/spnc-runtime.h"

namespace spnc_rt {

    spn_runtime* spn_runtime::_instance = nullptr;

    spn_runtime& spn_runtime::instance() {
      if(!_instance){
        _instance = new spn_runtime{};
      }
      return *_instance;
    }

    void spn_runtime::execute(const Kernel &kernel, size_t num_elements, void *inputs, double *outputs) {

    }

}
