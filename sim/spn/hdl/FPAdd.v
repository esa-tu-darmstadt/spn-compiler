module CompareComb( // @[:@3.2]
  input  [7:0] io_op1, // @[:@6.4]
  input  [7:0] io_op2, // @[:@6.4]
  output       io_gte // @[:@6.4]
);
  wire  _T_11; // @[FPAdd.scala 72:20:@8.4]
  assign _T_11 = io_op1 >= io_op2; // @[FPAdd.scala 72:20:@8.4]
  assign io_gte = _T_11;
endmodule
module Swap( // @[:@11.2]
  input         clock, // @[:@12.4]
  input  [22:0] io_in_op1_m, // @[:@14.4]
  input  [7:0]  io_in_op1_e, // @[:@14.4]
  input  [22:0] io_in_op2_m, // @[:@14.4]
  input  [7:0]  io_in_op2_e, // @[:@14.4]
  output [22:0] io_out_op1_m, // @[:@14.4]
  output [7:0]  io_out_op1_e, // @[:@14.4]
  output [22:0] io_out_op2_m, // @[:@14.4]
  output [7:0]  io_out_op2_e // @[:@14.4]
);
  wire [7:0] compare_io_op1; // @[FPAdd.scala 83:23:@16.4]
  wire [7:0] compare_io_op2; // @[FPAdd.scala 83:23:@16.4]
  wire  compare_io_gte; // @[FPAdd.scala 83:23:@16.4]
  reg [22:0] _T_7_m; // @[FPAdd.scala 87:24:@22.4]
  reg [31:0] _RAND_0;
  reg [7:0] _T_7_e; // @[FPAdd.scala 87:24:@22.4]
  reg [31:0] _RAND_1;
  reg [22:0] _T_10_m; // @[FPAdd.scala 88:24:@28.4]
  reg [31:0] _RAND_2;
  reg [7:0] _T_10_e; // @[FPAdd.scala 88:24:@28.4]
  reg [31:0] _RAND_3;
  wire [22:0] _T_5_m; // @[FPAdd.scala 87:28:@21.4]
  wire [7:0] _T_5_e; // @[FPAdd.scala 87:28:@21.4]
  wire [22:0] _T_8_m; // @[FPAdd.scala 88:28:@27.4]
  wire [7:0] _T_8_e; // @[FPAdd.scala 88:28:@27.4]
  CompareComb compare ( // @[FPAdd.scala 83:23:@16.4]
    .io_op1(compare_io_op1),
    .io_op2(compare_io_op2),
    .io_gte(compare_io_gte)
  );
  assign _T_5_m = compare_io_gte ? io_in_op1_m : io_in_op2_m; // @[FPAdd.scala 87:28:@21.4]
  assign _T_5_e = compare_io_gte ? io_in_op1_e : io_in_op2_e; // @[FPAdd.scala 87:28:@21.4]
  assign _T_8_m = compare_io_gte ? io_in_op2_m : io_in_op1_m; // @[FPAdd.scala 88:28:@27.4]
  assign _T_8_e = compare_io_gte ? io_in_op2_e : io_in_op1_e; // @[FPAdd.scala 88:28:@27.4]
  assign io_out_op1_m = _T_7_m;
  assign io_out_op1_e = _T_7_e;
  assign io_out_op2_m = _T_10_m;
  assign io_out_op2_e = _T_10_e;
  assign compare_io_op1 = io_in_op1_e;
  assign compare_io_op2 = io_in_op2_e;
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
  _T_7_m = _RAND_0[22:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{$random}};
  _T_7_e = _RAND_1[7:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {1{$random}};
  _T_10_m = _RAND_2[22:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{$random}};
  _T_10_e = _RAND_3[7:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if (compare_io_gte) begin
      _T_7_m <= io_in_op1_m;
    end else begin
      _T_7_m <= io_in_op2_m;
    end
    if (compare_io_gte) begin
      _T_7_e <= io_in_op1_e;
    end else begin
      _T_7_e <= io_in_op2_e;
    end
    if (compare_io_gte) begin
      _T_10_m <= io_in_op2_m;
    end else begin
      _T_10_m <= io_in_op1_m;
    end
    if (compare_io_gte) begin
      _T_10_e <= io_in_op2_e;
    end else begin
      _T_10_e <= io_in_op1_e;
    end
  end
endmodule
module SubtractStage( // @[:@34.2]
  input        clock, // @[:@35.4]
  input  [7:0] io_minuend, // @[:@37.4]
  input  [7:0] io_subtrahend, // @[:@37.4]
  output [7:0] io_difference, // @[:@37.4]
  output       io_minuend_is_zero // @[:@37.4]
);
  reg [7:0] difference; // @[FPAdd.scala 98:27:@42.4]
  reg [31:0] _RAND_0;
  reg  _T_20; // @[FPAdd.scala 100:32:@46.4]
  reg [31:0] _RAND_1;
  wire [8:0] _T_13; // @[FPAdd.scala 98:39:@39.4]
  wire [8:0] _T_14; // @[FPAdd.scala 98:39:@40.4]
  wire [7:0] _T_15; // @[FPAdd.scala 98:39:@41.4]
  wire  _T_18; // @[FPAdd.scala 100:44:@45.4]
  assign _T_13 = io_minuend - io_subtrahend; // @[FPAdd.scala 98:39:@39.4]
  assign _T_14 = $unsigned(_T_13); // @[FPAdd.scala 98:39:@40.4]
  assign _T_15 = _T_14[7:0]; // @[FPAdd.scala 98:39:@41.4]
  assign _T_18 = io_minuend == 8'h0; // @[FPAdd.scala 100:44:@45.4]
  assign io_difference = difference;
  assign io_minuend_is_zero = _T_20;
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
  difference = _RAND_0[7:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{$random}};
  _T_20 = _RAND_1[0:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    difference <= _T_15;
    _T_20 <= _T_18;
  end
endmodule
module ShiftStage( // @[:@50.2]
  input         clock, // @[:@51.4]
  input  [22:0] io_value, // @[:@53.4]
  input  [7:0]  io_shamt, // @[:@53.4]
  output [23:0] io_out // @[:@53.4]
);
  reg [23:0] shifted; // @[FPAdd.scala 109:24:@57.4]
  reg [31:0] _RAND_0;
  wire [23:0] _T_12; // @[Cat.scala 30:58:@55.4]
  wire [23:0] _T_13; // @[FPAdd.scala 109:44:@56.4]
  assign _T_12 = {1'h1,io_value}; // @[Cat.scala 30:58:@55.4]
  assign _T_13 = _T_12 >> io_shamt; // @[FPAdd.scala 109:44:@56.4]
  assign io_out = shifted;
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
  shifted = _RAND_0[23:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    shifted <= _T_13;
  end
endmodule
module PipelinedAdder( // @[:@61.2]
  input         clock, // @[:@62.4]
  input  [23:0] io_a, // @[:@64.4]
  input  [23:0] io_b, // @[:@64.4]
  output [24:0] io_r // @[:@64.4]
);
  reg [32:0] _T_17; // @[PipelinedAdder.scala 82:25:@73.4]
  reg [63:0] _RAND_0;
  wire [31:0] aPad; // @[Cat.scala 30:58:@66.4]
  wire [31:0] bPad; // @[Cat.scala 30:58:@67.4]
  wire [32:0] _T_13; // @[PipelinedAdder.scala 82:28:@70.4]
  wire [33:0] _T_14; // @[PipelinedAdder.scala 82:33:@71.4]
  wire [32:0] _T_15; // @[PipelinedAdder.scala 82:33:@72.4]
  wire [31:0] flushRes_0; // @[PipelinedAdder.scala 83:12:@75.4]
  wire  carrySignals_1; // @[PipelinedAdder.scala 83:38:@76.4]
  wire [32:0] _T_18; // @[Cat.scala 30:58:@77.4]
  assign aPad = {8'h0,io_a}; // @[Cat.scala 30:58:@66.4]
  assign bPad = {8'h0,io_b}; // @[Cat.scala 30:58:@67.4]
  assign _T_13 = aPad + bPad; // @[PipelinedAdder.scala 82:28:@70.4]
  assign _T_14 = _T_13 + 33'h0; // @[PipelinedAdder.scala 82:33:@71.4]
  assign _T_15 = _T_14[32:0]; // @[PipelinedAdder.scala 82:33:@72.4]
  assign flushRes_0 = _T_17[31:0]; // @[PipelinedAdder.scala 83:12:@75.4]
  assign carrySignals_1 = _T_17[32]; // @[PipelinedAdder.scala 83:38:@76.4]
  assign _T_18 = {carrySignals_1,flushRes_0}; // @[Cat.scala 30:58:@77.4]
  assign io_r = _T_18[24:0];
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
  _RAND_0 = {2{$random}};
  _T_17 = _RAND_0[32:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    _T_17 <= _T_15;
  end
endmodule
module MantissaShifterComb( // @[:@80.2]
  input  [24:0] io_sum, // @[:@83.4]
  output [22:0] io_out, // @[:@83.4]
  output        io_shifted // @[:@83.4]
);
  wire  _T_11; // @[FPAdd.scala 119:36:@85.4]
  wire [23:0] _T_13; // @[FPAdd.scala 120:16:@87.4]
  wire [22:0] _T_14; // @[FPAdd.scala 120:24:@88.4]
  wire [22:0] _T_15; // @[FPAdd.scala 121:16:@89.4]
  wire [22:0] _T_16; // @[FPAdd.scala 121:24:@90.4]
  wire [22:0] shifted_sum; // @[FPAdd.scala 119:24:@91.4]
  assign _T_11 = io_sum[24:24]; // @[FPAdd.scala 119:36:@85.4]
  assign _T_13 = io_sum[23:0]; // @[FPAdd.scala 120:16:@87.4]
  assign _T_14 = _T_13[23:1]; // @[FPAdd.scala 120:24:@88.4]
  assign _T_15 = io_sum[22:0]; // @[FPAdd.scala 121:16:@89.4]
  assign _T_16 = _T_15[22:0]; // @[FPAdd.scala 121:24:@90.4]
  assign shifted_sum = _T_11 ? _T_14 : _T_16; // @[FPAdd.scala 119:24:@91.4]
  assign io_out = shifted_sum;
  assign io_shifted = _T_11;
endmodule
module ExponentAdder( // @[:@96.2]
  input        clock, // @[:@97.4]
  input  [7:0] io_e_in, // @[:@99.4]
  input        io_add, // @[:@99.4]
  input        io_inputs_are_zero, // @[:@99.4]
  output [7:0] io_e_out // @[:@99.4]
);
  reg [7:0] out; // @[FPAdd.scala 134:20:@106.4]
  reg [31:0] _RAND_0;
  wire  _T_14; // @[FPAdd.scala 134:35:@101.4]
  wire  _T_15; // @[FPAdd.scala 134:32:@102.4]
  wire [8:0] _T_17; // @[FPAdd.scala 134:64:@103.4]
  wire [7:0] _T_18; // @[FPAdd.scala 134:64:@104.4]
  wire [7:0] _T_19; // @[FPAdd.scala 134:24:@105.4]
  assign _T_14 = io_inputs_are_zero == 1'h0; // @[FPAdd.scala 134:35:@101.4]
  assign _T_15 = io_add & _T_14; // @[FPAdd.scala 134:32:@102.4]
  assign _T_17 = io_e_in + 8'h1; // @[FPAdd.scala 134:64:@103.4]
  assign _T_18 = _T_17[7:0]; // @[FPAdd.scala 134:64:@104.4]
  assign _T_19 = _T_15 ? _T_18 : io_e_in; // @[FPAdd.scala 134:24:@105.4]
  assign io_e_out = out;
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
  out = _RAND_0[7:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if (_T_15) begin
      out <= _T_18;
    end else begin
      out <= io_e_in;
    end
  end
endmodule
module FPAdd( // @[:@110.2]
  input         clock, // @[:@111.4]
  input         reset, // @[:@112.4]
  input  [30:0] io_a, // @[:@113.4]
  input  [30:0] io_b, // @[:@113.4]
  output [30:0] io_r // @[:@113.4]
);
  reg [30:0] op1_packed; // @[FPAdd.scala 10:27:@115.4]
  reg [31:0] _RAND_0;
  reg [30:0] op2_packed; // @[FPAdd.scala 11:27:@117.4]
  reg [31:0] _RAND_1;
  wire  swap_clock; // @[FPAdd.scala 17:20:@129.4]
  wire [22:0] swap_io_in_op1_m; // @[FPAdd.scala 17:20:@129.4]
  wire [7:0] swap_io_in_op1_e; // @[FPAdd.scala 17:20:@129.4]
  wire [22:0] swap_io_in_op2_m; // @[FPAdd.scala 17:20:@129.4]
  wire [7:0] swap_io_in_op2_e; // @[FPAdd.scala 17:20:@129.4]
  wire [22:0] swap_io_out_op1_m; // @[FPAdd.scala 17:20:@129.4]
  wire [7:0] swap_io_out_op1_e; // @[FPAdd.scala 17:20:@129.4]
  wire [22:0] swap_io_out_op2_m; // @[FPAdd.scala 17:20:@129.4]
  wire [7:0] swap_io_out_op2_e; // @[FPAdd.scala 17:20:@129.4]
  wire  subtractStage_clock; // @[FPAdd.scala 22:29:@136.4]
  wire [7:0] subtractStage_io_minuend; // @[FPAdd.scala 22:29:@136.4]
  wire [7:0] subtractStage_io_subtrahend; // @[FPAdd.scala 22:29:@136.4]
  wire [7:0] subtractStage_io_difference; // @[FPAdd.scala 22:29:@136.4]
  wire  subtractStage_io_minuend_is_zero; // @[FPAdd.scala 22:29:@136.4]
  reg [22:0] m1_2; // @[FPAdd.scala 26:21:@141.4]
  reg [31:0] _RAND_2;
  reg [22:0] m2_2; // @[FPAdd.scala 27:21:@143.4]
  reg [31:0] _RAND_3;
  reg [7:0] e1_2; // @[FPAdd.scala 28:21:@145.4]
  reg [31:0] _RAND_4;
  wire  shiftStage_clock; // @[FPAdd.scala 32:26:@149.4]
  wire [22:0] shiftStage_io_value; // @[FPAdd.scala 32:26:@149.4]
  wire [7:0] shiftStage_io_shamt; // @[FPAdd.scala 32:26:@149.4]
  wire [23:0] shiftStage_io_out; // @[FPAdd.scala 32:26:@149.4]
  reg [22:0] m1_3; // @[FPAdd.scala 35:21:@154.4]
  reg [31:0] _RAND_5;
  reg [7:0] e1_3; // @[FPAdd.scala 37:21:@156.4]
  reg [31:0] _RAND_6;
  reg  minuend_is_zero_3; // @[FPAdd.scala 38:34:@158.4]
  reg [31:0] _RAND_7;
  wire  PipelinedAdder_clock; // @[PipelinedAdder.scala 98:21:@161.4]
  wire [23:0] PipelinedAdder_io_a; // @[PipelinedAdder.scala 98:21:@161.4]
  wire [23:0] PipelinedAdder_io_b; // @[PipelinedAdder.scala 98:21:@161.4]
  wire [24:0] PipelinedAdder_io_r; // @[PipelinedAdder.scala 98:21:@161.4]
  reg  minuend_is_zero_4; // @[Reg.scala 11:16:@166.4]
  reg [31:0] _RAND_8;
  reg [7:0] e1_4; // @[Reg.scala 11:16:@170.4]
  reg [31:0] _RAND_9;
  wire [24:0] mantissaShifterComb_io_sum; // @[FPAdd.scala 48:35:@174.4]
  wire [22:0] mantissaShifterComb_io_out; // @[FPAdd.scala 48:35:@174.4]
  wire  mantissaShifterComb_io_shifted; // @[FPAdd.scala 48:35:@174.4]
  wire  exponentAdder_clock; // @[FPAdd.scala 52:29:@178.4]
  wire [7:0] exponentAdder_io_e_in; // @[FPAdd.scala 52:29:@178.4]
  wire  exponentAdder_io_add; // @[FPAdd.scala 52:29:@178.4]
  wire  exponentAdder_io_inputs_are_zero; // @[FPAdd.scala 52:29:@178.4]
  wire [7:0] exponentAdder_io_e_out; // @[FPAdd.scala 52:29:@178.4]
  reg [22:0] m_6; // @[FPAdd.scala 56:20:@184.4]
  reg [31:0] _RAND_10;
  wire [7:0] op1_e; // @[UFloat.scala 19:26:@120.4]
  wire [22:0] op1_m; // @[UFloat.scala 20:26:@122.4]
  wire [7:0] op2_e; // @[UFloat.scala 19:26:@125.4]
  wire [22:0] op2_m; // @[UFloat.scala 20:26:@127.4]
  wire [23:0] a; // @[Cat.scala 30:58:@160.4]
  wire [7:0] out_e; // @[FPAdd.scala 59:17:@186.4]
  wire [30:0] output_together; // @[Cat.scala 30:58:@189.4]
  Swap swap ( // @[FPAdd.scala 17:20:@129.4]
    .clock(swap_clock),
    .io_in_op1_m(swap_io_in_op1_m),
    .io_in_op1_e(swap_io_in_op1_e),
    .io_in_op2_m(swap_io_in_op2_m),
    .io_in_op2_e(swap_io_in_op2_e),
    .io_out_op1_m(swap_io_out_op1_m),
    .io_out_op1_e(swap_io_out_op1_e),
    .io_out_op2_m(swap_io_out_op2_m),
    .io_out_op2_e(swap_io_out_op2_e)
  );
  SubtractStage subtractStage ( // @[FPAdd.scala 22:29:@136.4]
    .clock(subtractStage_clock),
    .io_minuend(subtractStage_io_minuend),
    .io_subtrahend(subtractStage_io_subtrahend),
    .io_difference(subtractStage_io_difference),
    .io_minuend_is_zero(subtractStage_io_minuend_is_zero)
  );
  ShiftStage shiftStage ( // @[FPAdd.scala 32:26:@149.4]
    .clock(shiftStage_clock),
    .io_value(shiftStage_io_value),
    .io_shamt(shiftStage_io_shamt),
    .io_out(shiftStage_io_out)
  );
  PipelinedAdder PipelinedAdder ( // @[PipelinedAdder.scala 98:21:@161.4]
    .clock(PipelinedAdder_clock),
    .io_a(PipelinedAdder_io_a),
    .io_b(PipelinedAdder_io_b),
    .io_r(PipelinedAdder_io_r)
  );
  MantissaShifterComb mantissaShifterComb ( // @[FPAdd.scala 48:35:@174.4]
    .io_sum(mantissaShifterComb_io_sum),
    .io_out(mantissaShifterComb_io_out),
    .io_shifted(mantissaShifterComb_io_shifted)
  );
  ExponentAdder exponentAdder ( // @[FPAdd.scala 52:29:@178.4]
    .clock(exponentAdder_clock),
    .io_e_in(exponentAdder_io_e_in),
    .io_add(exponentAdder_io_add),
    .io_inputs_are_zero(exponentAdder_io_inputs_are_zero),
    .io_e_out(exponentAdder_io_e_out)
  );
  assign op1_e = op1_packed[30:23]; // @[UFloat.scala 19:26:@120.4]
  assign op1_m = op1_packed[22:0]; // @[UFloat.scala 20:26:@122.4]
  assign op2_e = op2_packed[30:23]; // @[UFloat.scala 19:26:@125.4]
  assign op2_m = op2_packed[22:0]; // @[UFloat.scala 20:26:@127.4]
  assign a = {1'h1,m1_3}; // @[Cat.scala 30:58:@160.4]
  assign out_e = exponentAdder_io_e_out; // @[FPAdd.scala 59:17:@186.4]
  assign output_together = {out_e,m_6}; // @[Cat.scala 30:58:@189.4]
  assign io_r = output_together;
  assign swap_clock = clock;
  assign swap_io_in_op1_m = op1_m;
  assign swap_io_in_op1_e = op1_e;
  assign swap_io_in_op2_m = op2_m;
  assign swap_io_in_op2_e = op2_e;
  assign subtractStage_clock = clock;
  assign subtractStage_io_minuend = swap_io_out_op1_e;
  assign subtractStage_io_subtrahend = swap_io_out_op2_e;
  assign shiftStage_clock = clock;
  assign shiftStage_io_value = m2_2;
  assign shiftStage_io_shamt = subtractStage_io_difference;
  assign PipelinedAdder_clock = clock;
  assign PipelinedAdder_io_a = a;
  assign PipelinedAdder_io_b = shiftStage_io_out;
  assign mantissaShifterComb_io_sum = PipelinedAdder_io_r;
  assign exponentAdder_clock = clock;
  assign exponentAdder_io_e_in = e1_4;
  assign exponentAdder_io_add = mantissaShifterComb_io_shifted;
  assign exponentAdder_io_inputs_are_zero = minuend_is_zero_4;
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
  op1_packed = _RAND_0[30:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{$random}};
  op2_packed = _RAND_1[30:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {1{$random}};
  m1_2 = _RAND_2[22:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{$random}};
  m2_2 = _RAND_3[22:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_4 = {1{$random}};
  e1_2 = _RAND_4[7:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_5 = {1{$random}};
  m1_3 = _RAND_5[22:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_6 = {1{$random}};
  e1_3 = _RAND_6[7:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_7 = {1{$random}};
  minuend_is_zero_3 = _RAND_7[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_8 = {1{$random}};
  minuend_is_zero_4 = _RAND_8[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_9 = {1{$random}};
  e1_4 = _RAND_9[7:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_10 = {1{$random}};
  m_6 = _RAND_10[22:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    op1_packed <= io_a;
    op2_packed <= io_b;
    m1_2 <= swap_io_out_op1_m;
    m2_2 <= swap_io_out_op2_m;
    e1_2 <= swap_io_out_op1_e;
    m1_3 <= m1_2;
    e1_3 <= e1_2;
    minuend_is_zero_3 <= subtractStage_io_minuend_is_zero;
    minuend_is_zero_4 <= minuend_is_zero_3;
    e1_4 <= e1_3;
    m_6 <= mantissaShifterComb_io_out;
  end
endmodule
