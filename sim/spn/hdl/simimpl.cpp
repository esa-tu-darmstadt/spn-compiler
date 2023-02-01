
uint8_t PySPNSim::convertIndex(uint32_t input) {
  uint8_t result = static_cast<uint8_t>(input);
  return result;
}

double PySPNSim::convertProb(uint32_t prob) {
  uint64_t p = prob;
  return *reinterpret_cast<const float *>(&p);
}
  
void PySPNSim::setInput(const std::vector<uint32_t>& input) {
  assert(input.size() == 5);

  top->in_0 = convertIndex(input[0]);
  top->in_1 = convertIndex(input[1]);
  top->in_2 = convertIndex(input[2]);
  top->in_3 = convertIndex(input[3]);
  top->in_4 = convertIndex(input[4]);

}
  