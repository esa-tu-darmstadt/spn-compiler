module FP2DoubleConverter( // @[:@3.2]
  input         clock, // @[:@4.4]
  input         reset, // @[:@5.4]
  input  [30:0] io_op, // @[:@6.4]
  output [63:0] io_r // @[:@6.4]
);
  reg [7:0] inputs_e; // @[FP2DoubleConverter.scala 10:22:@10.4]
  reg [31:0] _RAND_0;
  reg [22:0] inputs_m; // @[FP2DoubleConverter.scala 11:22:@14.4]
  reg [31:0] _RAND_1;
  reg [51:0] res_m; // @[FP2DoubleConverter.scala 27:20:@26.4]
  reg [63:0] _RAND_2;
  reg [10:0] res_e; // @[FP2DoubleConverter.scala 27:20:@26.4]
  reg [31:0] _RAND_3;
  wire [7:0] _T_10; // @[FP2DoubleConverter.scala 10:33:@9.4]
  wire [22:0] _T_13; // @[FP2DoubleConverter.scala 11:33:@13.4]
  wire [51:0] _GEN_0; // @[FP2DoubleConverter.scala 19:31:@17.4]
  wire [51:0] new_mantissa; // @[FP2DoubleConverter.scala 19:31:@17.4]
  wire [9:0] _GEN_1; // @[FP2DoubleConverter.scala 20:39:@18.4]
  wire [10:0] new_exponent_nonzero; // @[FP2DoubleConverter.scala 20:39:@18.4]
  wire  _T_18; // @[FP2DoubleConverter.scala 21:30:@19.4]
  wire  _T_20; // @[FP2DoubleConverter.scala 21:50:@20.4]
  wire  inputs_zero; // @[FP2DoubleConverter.scala 21:38:@21.4]
  wire [10:0] new_exponent; // @[FP2DoubleConverter.scala 22:25:@22.4]
  wire [62:0] _T_24; // @[Cat.scala 30:58:@29.4]
  assign _T_10 = io_op[30:23]; // @[FP2DoubleConverter.scala 10:33:@9.4]
  assign _T_13 = io_op[22:0]; // @[FP2DoubleConverter.scala 11:33:@13.4]
  assign _GEN_0 = {{29'd0}, inputs_m}; // @[FP2DoubleConverter.scala 19:31:@17.4]
  assign new_mantissa = _GEN_0 << 29; // @[FP2DoubleConverter.scala 19:31:@17.4]
  assign _GEN_1 = {{2'd0}, inputs_e}; // @[FP2DoubleConverter.scala 20:39:@18.4]
  assign new_exponent_nonzero = _GEN_1 + 10'h380; // @[FP2DoubleConverter.scala 20:39:@18.4]
  assign _T_18 = inputs_e == 8'h0; // @[FP2DoubleConverter.scala 21:30:@19.4]
  assign _T_20 = inputs_m == 23'h0; // @[FP2DoubleConverter.scala 21:50:@20.4]
  assign inputs_zero = _T_18 & _T_20; // @[FP2DoubleConverter.scala 21:38:@21.4]
  assign new_exponent = inputs_zero ? 11'h0 : new_exponent_nonzero; // @[FP2DoubleConverter.scala 22:25:@22.4]
  assign _T_24 = {res_e,res_m}; // @[Cat.scala 30:58:@29.4]
  assign io_r = {{1'd0}, _T_24};
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE
  integer initvar;
  initial begin
    `ifndef verilator
      #0.002 begin end
    `endif
  `ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{$random}};
  inputs_e = _RAND_0[7:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{$random}};
  inputs_m = _RAND_1[22:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {2{$random}};
  res_m = _RAND_2[51:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{$random}};
  res_e = _RAND_3[10:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    inputs_e <= _T_10;
    inputs_m <= _T_13;
    res_m <= new_mantissa;
    if (inputs_zero) begin
      res_e <= 11'h0;
    end else begin
      res_e <= new_exponent_nonzero;
    end
  end
endmodule
