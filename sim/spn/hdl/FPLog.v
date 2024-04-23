module FPLog(
  input         clk,
  input         rst,
  input  [30:0] in_a,
  output [30:0] out_b
);
  assign out_b = in_a;
endmodule