#pragma once

#include <cstdlib>
#include <cstdint>
#include <bitpacker/bitpacker.hpp>


namespace spnc_rt::tapasco_wrapper {

template <class T>
T divRound(const T& a, const T& b) {
  return a / b + (a % b ? 1 : 0);
}

template <class InputIt>
void pack(InputIt inBegin, InputIt inEnd, size_t bitSize, std::vector<uint8_t>& buf) {
  size_t inSize = std::distance(inBegin, inEnd);
  buf.resize(divRound<size_t>(inSize * bitSize, 8));

  for (size_t i = 0; i < inSize; ++i)
    bitpacker::insert(
      buf,
      i * bitSize,
      bitSize,
      *(inBegin + i)
    );
}

}