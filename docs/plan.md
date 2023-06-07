# Implementation
- Create .fir files of `IPECLoadUnit` and `IPECStoreUnit` with the usual bit widths and put them into this project.
- Implement a AXI4Lite register file.
- Tie everything together into a top level module akin to `AXI4StreamMapper`.
  - Load and parse the correct .fir files.
- Build a top level `CocoTb` simulation.
- Adapt the `CreateVivadoProject` stage.

# Evaluation
- Compare the resource usage to `xspn-fpga`.
- Test on real hardware. (if possible)