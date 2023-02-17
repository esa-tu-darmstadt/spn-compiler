#pragma once

#include <cstdint>


namespace spnc {

struct ControllerDescription {
  uint32_t bodyDelay;
  uint32_t fifoDepth;
  // these can be weird numbers like 31 bits
  uint32_t spnVarCount;
  uint32_t spnBitsPerVar;
  uint32_t spnResultWidth;

  // sets the width for S_AXIS_CONTROLLER and M_AXIS_CONTROLLER
  // + sets the widths of the SPNController input/output AXIStreams
  uint32_t mAxisControllerWidth;
  uint32_t sAxisControllerWidth;

  // sets the width for S_AXIS and M_AXIS and also M_AXI
  uint32_t memDataWidth;
  uint32_t memAddrWidth;

  // sets the width for S_AXI_LITE
  uint32_t liteDataWidth;
  uint32_t liteAddrWidth;
};

}