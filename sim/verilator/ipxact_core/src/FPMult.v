module DSPMult24x17( // @[:@3.2]
  input         clock, // @[:@4.4]
  input  [23:0] io_a, // @[:@6.4]
  input  [16:0] io_b, // @[:@6.4]
  output [40:0] io_r // @[:@6.4]
);
  reg [40:0] rOut; // @[DSPMult.scala 70:21:@9.4]
  reg [63:0] _RAND_0;
  wire [23:0] _GEN_0; // @[DSPMult.scala 70:27:@8.4]
  wire [40:0] _T_11; // @[DSPMult.scala 70:27:@8.4]
  assign _GEN_0 = {{7'd0}, io_b}; // @[DSPMult.scala 70:27:@8.4]
  assign _T_11 = io_a * _GEN_0; // @[DSPMult.scala 70:27:@8.4]
  assign io_r = rOut;
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
  rOut = _RAND_0[40:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    rOut <= _T_11;
  end
endmodule
module PipelinedAdder( // @[:@23.2]
  input         clock, // @[:@24.4]
  input  [47:0] io_a, // @[:@26.4]
  input  [47:0] io_b, // @[:@26.4]
  output [48:0] io_r // @[:@26.4]
);
  reg [31:0] aDelayed_1; // @[Reg.scala 11:16:@34.4]
  reg [31:0] _RAND_0;
  reg [31:0] bDelayed_1; // @[Reg.scala 11:16:@38.4]
  reg [31:0] _RAND_1;
  reg [32:0] _T_21; // @[PipelinedAdder.scala 82:25:@45.4]
  reg [63:0] _RAND_2;
  reg [32:0] _T_26; // @[PipelinedAdder.scala 82:25:@52.4]
  reg [63:0] _RAND_3;
  reg [31:0] flushRes_0; // @[Reg.scala 11:16:@56.4]
  reg [31:0] _RAND_4;
  wire [63:0] aPad; // @[Cat.scala 30:58:@28.4]
  wire [63:0] bPad; // @[Cat.scala 30:58:@29.4]
  wire [31:0] a_0; // @[PipelinedAdder.scala 39:53:@30.4]
  wire [31:0] a_1; // @[PipelinedAdder.scala 39:53:@31.4]
  wire [31:0] b_0; // @[PipelinedAdder.scala 40:53:@32.4]
  wire [31:0] b_1; // @[PipelinedAdder.scala 40:53:@33.4]
  wire [32:0] _T_17; // @[PipelinedAdder.scala 82:28:@42.4]
  wire [33:0] _T_18; // @[PipelinedAdder.scala 82:33:@43.4]
  wire [32:0] _T_19; // @[PipelinedAdder.scala 82:33:@44.4]
  wire [31:0] resultSignals_0; // @[PipelinedAdder.scala 83:12:@47.4]
  wire  carrySignals_1; // @[PipelinedAdder.scala 83:38:@48.4]
  wire [32:0] _T_22; // @[PipelinedAdder.scala 82:28:@49.4]
  wire [32:0] _GEN_3; // @[PipelinedAdder.scala 82:33:@50.4]
  wire [33:0] _T_23; // @[PipelinedAdder.scala 82:33:@50.4]
  wire [32:0] _T_24; // @[PipelinedAdder.scala 82:33:@51.4]
  wire [31:0] flushRes_1; // @[PipelinedAdder.scala 83:12:@54.4]
  wire  carrySignals_2; // @[PipelinedAdder.scala 83:38:@55.4]
  wire [63:0] _T_29; // @[Cat.scala 30:58:@60.4]
  wire [64:0] _T_30; // @[Cat.scala 30:58:@61.4]
  assign aPad = {16'h0,io_a}; // @[Cat.scala 30:58:@28.4]
  assign bPad = {16'h0,io_b}; // @[Cat.scala 30:58:@29.4]
  assign a_0 = aPad[31:0]; // @[PipelinedAdder.scala 39:53:@30.4]
  assign a_1 = aPad[63:32]; // @[PipelinedAdder.scala 39:53:@31.4]
  assign b_0 = bPad[31:0]; // @[PipelinedAdder.scala 40:53:@32.4]
  assign b_1 = bPad[63:32]; // @[PipelinedAdder.scala 40:53:@33.4]
  assign _T_17 = a_0 + b_0; // @[PipelinedAdder.scala 82:28:@42.4]
  assign _T_18 = _T_17 + 33'h0; // @[PipelinedAdder.scala 82:33:@43.4]
  assign _T_19 = _T_18[32:0]; // @[PipelinedAdder.scala 82:33:@44.4]
  assign resultSignals_0 = _T_21[31:0]; // @[PipelinedAdder.scala 83:12:@47.4]
  assign carrySignals_1 = _T_21[32]; // @[PipelinedAdder.scala 83:38:@48.4]
  assign _T_22 = aDelayed_1 + bDelayed_1; // @[PipelinedAdder.scala 82:28:@49.4]
  assign _GEN_3 = {{32'd0}, carrySignals_1}; // @[PipelinedAdder.scala 82:33:@50.4]
  assign _T_23 = _T_22 + _GEN_3; // @[PipelinedAdder.scala 82:33:@50.4]
  assign _T_24 = _T_23[32:0]; // @[PipelinedAdder.scala 82:33:@51.4]
  assign flushRes_1 = _T_26[31:0]; // @[PipelinedAdder.scala 83:12:@54.4]
  assign carrySignals_2 = _T_26[32]; // @[PipelinedAdder.scala 83:38:@55.4]
  assign _T_29 = {flushRes_1,flushRes_0}; // @[Cat.scala 30:58:@60.4]
  assign _T_30 = {carrySignals_2,_T_29}; // @[Cat.scala 30:58:@61.4]
  assign io_r = _T_30[48:0];
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
  aDelayed_1 = _RAND_0[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{$random}};
  bDelayed_1 = _RAND_1[31:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_2 = {2{$random}};
  _T_21 = _RAND_2[32:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {2{$random}};
  _T_26 = _RAND_3[32:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_4 = {1{$random}};
  flushRes_0 = _RAND_4[31:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    aDelayed_1 <= a_1;
    bDelayed_1 <= b_1;
    _T_21 <= _T_19;
    _T_26 <= _T_24;
    flushRes_0 <= resultSignals_0;
  end
endmodule
module DSPMult( // @[:@64.2]
  input         clock, // @[:@65.4]
  input  [23:0] io_a, // @[:@67.4]
  input  [23:0] io_b, // @[:@67.4]
  output [47:0] io_r // @[:@67.4]
);
  wire  DSPMult24x17_clock; // @[DSPMult.scala 178:22:@71.4]
  wire [23:0] DSPMult24x17_io_a; // @[DSPMult.scala 178:22:@71.4]
  wire [16:0] DSPMult24x17_io_b; // @[DSPMult.scala 178:22:@71.4]
  wire [40:0] DSPMult24x17_io_r; // @[DSPMult.scala 178:22:@71.4]
  wire  DSPMult24x17_1_clock; // @[DSPMult.scala 178:22:@79.4]
  wire [23:0] DSPMult24x17_1_io_a; // @[DSPMult.scala 178:22:@79.4]
  wire [16:0] DSPMult24x17_1_io_b; // @[DSPMult.scala 178:22:@79.4]
  wire [40:0] DSPMult24x17_1_io_r; // @[DSPMult.scala 178:22:@79.4]
  wire  PipelinedAdder_clock; // @[PipelinedAdder.scala 98:21:@84.4]
  wire [47:0] PipelinedAdder_io_a; // @[PipelinedAdder.scala 98:21:@84.4]
  wire [47:0] PipelinedAdder_io_b; // @[PipelinedAdder.scala 98:21:@84.4]
  wire [48:0] PipelinedAdder_io_r; // @[PipelinedAdder.scala 98:21:@84.4]
  wire [6:0] _T_12; // @[DSPMult.scala 114:16:@70.4]
  wire [57:0] _T_14; // @[Cat.scala 30:58:@76.4]
  wire [16:0] _T_16; // @[DSPMult.scala 114:16:@78.4]
  DSPMult24x17 DSPMult24x17 ( // @[DSPMult.scala 178:22:@71.4]
    .clock(DSPMult24x17_clock),
    .io_a(DSPMult24x17_io_a),
    .io_b(DSPMult24x17_io_b),
    .io_r(DSPMult24x17_io_r)
  );
  DSPMult24x17 DSPMult24x17_1 ( // @[DSPMult.scala 178:22:@79.4]
    .clock(DSPMult24x17_1_clock),
    .io_a(DSPMult24x17_1_io_a),
    .io_b(DSPMult24x17_1_io_b),
    .io_r(DSPMult24x17_1_io_r)
  );
  PipelinedAdder PipelinedAdder ( // @[PipelinedAdder.scala 98:21:@84.4]
    .clock(PipelinedAdder_clock),
    .io_a(PipelinedAdder_io_a),
    .io_b(PipelinedAdder_io_b),
    .io_r(PipelinedAdder_io_r)
  );
  assign _T_12 = io_b[23:17]; // @[DSPMult.scala 114:16:@70.4]
  assign _T_14 = {DSPMult24x17_io_r,17'h0}; // @[Cat.scala 30:58:@76.4]
  assign _T_16 = io_b[16:0]; // @[DSPMult.scala 114:16:@78.4]
  assign io_r = PipelinedAdder_io_r[47:0];
  assign DSPMult24x17_clock = clock;
  assign DSPMult24x17_io_a = io_a;
  assign DSPMult24x17_io_b = {{10'd0}, _T_12};
  assign DSPMult24x17_1_clock = clock;
  assign DSPMult24x17_1_io_a = io_a;
  assign DSPMult24x17_1_io_b = _T_16;
  assign PipelinedAdder_clock = clock;
  assign PipelinedAdder_io_a = _T_14[47:0];
  assign PipelinedAdder_io_b = {{7'd0}, DSPMult24x17_1_io_r};
endmodule
module ZeroCheckComb( // @[:@91.2]
  input  [22:0] io_op1_m, // @[:@94.4]
  input  [7:0]  io_op1_e, // @[:@94.4]
  input  [22:0] io_op2_m, // @[:@94.4]
  input  [7:0]  io_op2_e, // @[:@94.4]
  output        io_out // @[:@94.4]
);
  wire  _T_12; // @[FPMult.scala 53:25:@96.4]
  wire  _T_14; // @[FPMult.scala 53:45:@97.4]
  wire  _T_15; // @[FPMult.scala 53:33:@98.4]
  wire  _T_17; // @[FPMult.scala 53:65:@99.4]
  wire  _T_19; // @[FPMult.scala 53:85:@100.4]
  wire  _T_20; // @[FPMult.scala 53:73:@101.4]
  wire  output$; // @[FPMult.scala 53:53:@102.4]
  assign _T_12 = io_op1_m == 23'h0; // @[FPMult.scala 53:25:@96.4]
  assign _T_14 = io_op1_e == 8'h0; // @[FPMult.scala 53:45:@97.4]
  assign _T_15 = _T_12 & _T_14; // @[FPMult.scala 53:33:@98.4]
  assign _T_17 = io_op2_m == 23'h0; // @[FPMult.scala 53:65:@99.4]
  assign _T_19 = io_op2_e == 8'h0; // @[FPMult.scala 53:85:@100.4]
  assign _T_20 = _T_17 & _T_19; // @[FPMult.scala 53:73:@101.4]
  assign output$ = _T_15 | _T_20; // @[FPMult.scala 53:53:@102.4]
  assign io_out = output$;
endmodule
module AddExponentsComb( // @[:@105.2]
  input        clock, // @[:@106.4]
  input  [7:0] io_e1, // @[:@108.4]
  input  [7:0] io_e2, // @[:@108.4]
  output [8:0] io_eout // @[:@108.4]
);
  reg [8:0] out; // @[FPMult.scala 63:20:@111.4]
  reg [31:0] _RAND_0;
  wire [8:0] _T_11; // @[FPMult.scala 63:27:@110.4]
  assign _T_11 = io_e1 + io_e2; // @[FPMult.scala 63:27:@110.4]
  assign io_eout = out;
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
  out = _RAND_0[8:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    out <= _T_11;
  end
endmodule
module AddOneToExpAndSubtractOffsetComb( // @[:@115.2]
  input  [8:0] io_e1, // @[:@118.4]
  input        io_add, // @[:@118.4]
  output [7:0] io_e_out, // @[:@118.4]
  output       io_underflow_out // @[:@118.4]
);
  wire [9:0] _T_14; // @[Cat.scala 30:58:@120.4]
  wire [9:0] e1_s; // @[FPMult.scala 76:36:@121.4]
  wire [10:0] _T_17; // @[FPMult.scala 77:51:@123.4]
  wire [9:0] _T_18; // @[FPMult.scala 77:51:@124.4]
  wire [9:0] _T_19; // @[FPMult.scala 77:51:@125.4]
  wire [10:0] _T_21; // @[FPMult.scala 77:72:@126.4]
  wire [9:0] _T_22; // @[FPMult.scala 77:72:@127.4]
  wire [9:0] _T_23; // @[FPMult.scala 77:72:@128.4]
  wire [9:0] selected; // @[FPMult.scala 77:28:@129.4]
  wire  _T_24; // @[FPMult.scala 78:36:@130.4]
  wire [7:0] _T_25; // @[FPMult.scala 79:28:@132.4]
  assign _T_14 = {1'h0,io_e1}; // @[Cat.scala 30:58:@120.4]
  assign e1_s = $signed(_T_14); // @[FPMult.scala 76:36:@121.4]
  assign _T_17 = $signed(e1_s) - $signed(10'sh7e); // @[FPMult.scala 77:51:@123.4]
  assign _T_18 = _T_17[9:0]; // @[FPMult.scala 77:51:@124.4]
  assign _T_19 = $signed(_T_18); // @[FPMult.scala 77:51:@125.4]
  assign _T_21 = $signed(e1_s) - $signed(10'sh7f); // @[FPMult.scala 77:72:@126.4]
  assign _T_22 = _T_21[9:0]; // @[FPMult.scala 77:72:@127.4]
  assign _T_23 = $signed(_T_22); // @[FPMult.scala 77:72:@128.4]
  assign selected = io_add ? $signed(_T_19) : $signed(_T_23); // @[FPMult.scala 77:28:@129.4]
  assign _T_24 = selected[9:9]; // @[FPMult.scala 78:36:@130.4]
  assign _T_25 = selected[7:0]; // @[FPMult.scala 79:28:@132.4]
  assign io_e_out = _T_25;
  assign io_underflow_out = _T_24;
endmodule
module MantissaSelectorComb( // @[:@135.2]
  input  [47:0] io_m_in, // @[:@138.4]
  output [22:0] io_m_out // @[:@138.4]
);
  wire  _T_9; // @[FPMult.scala 87:34:@140.4]
  wire [46:0] _T_11; // @[FPMult.scala 88:34:@142.4]
  wire [22:0] _T_12; // @[FPMult.scala 88:42:@143.4]
  wire [45:0] _T_13; // @[FPMult.scala 89:34:@144.4]
  wire [22:0] _T_14; // @[FPMult.scala 89:42:@145.4]
  wire [22:0] out; // @[FPMult.scala 87:21:@146.4]
  assign _T_9 = io_m_in[47:47]; // @[FPMult.scala 87:34:@140.4]
  assign _T_11 = io_m_in[46:0]; // @[FPMult.scala 88:34:@142.4]
  assign _T_12 = _T_11[46:24]; // @[FPMult.scala 88:42:@143.4]
  assign _T_13 = io_m_in[45:0]; // @[FPMult.scala 89:34:@144.4]
  assign _T_14 = _T_13[45:23]; // @[FPMult.scala 89:42:@145.4]
  assign out = _T_9 ? _T_12 : _T_14; // @[FPMult.scala 87:21:@146.4]
  assign io_m_out = out;
endmodule
module ZeroHandler( // @[:@149.2]
  input         clock, // @[:@150.4]
  input  [22:0] io_m_in, // @[:@152.4]
  input  [7:0]  io_e_in, // @[:@152.4]
  input         io_underflow_in, // @[:@152.4]
  input         io_zero, // @[:@152.4]
  output [22:0] io_out_m, // @[:@152.4]
  output [7:0]  io_out_e // @[:@152.4]
);
  reg [22:0] m_out; // @[FPMult.scala 102:22:@156.4]
  reg [31:0] _RAND_0;
  reg [7:0] e_out; // @[FPMult.scala 103:22:@160.4]
  reg [31:0] _RAND_1;
  wire  _T_15; // @[FPMult.scala 102:35:@154.4]
  wire [22:0] _T_17; // @[FPMult.scala 102:26:@155.4]
  wire [7:0] _T_21; // @[FPMult.scala 103:26:@159.4]
  assign _T_15 = io_zero | io_underflow_in; // @[FPMult.scala 102:35:@154.4]
  assign _T_17 = _T_15 ? 23'h0 : io_m_in; // @[FPMult.scala 102:26:@155.4]
  assign _T_21 = _T_15 ? 8'h0 : io_e_in; // @[FPMult.scala 103:26:@159.4]
  assign io_out_m = m_out;
  assign io_out_e = e_out;
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
  m_out = _RAND_0[22:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{$random}};
  e_out = _RAND_1[7:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    if (_T_15) begin
      m_out <= 23'h0;
    end else begin
      m_out <= io_m_in;
    end
    if (_T_15) begin
      e_out <= 8'h0;
    end else begin
      e_out <= io_e_in;
    end
  end
endmodule
module FPMult( // @[:@165.2]
  input         clock, // @[:@166.4]
  input         reset, // @[:@167.4]
  input  [30:0] io_a, // @[:@168.4]
  input  [30:0] io_b, // @[:@168.4]
  output [30:0] io_r // @[:@168.4]
);
  reg [30:0] op1_packed; // @[FPMult.scala 11:27:@170.4]
  reg [31:0] _RAND_0;
  reg [30:0] op2_packed; // @[FPMult.scala 12:27:@172.4]
  reg [31:0] _RAND_1;
  wire  multiplyMantissas_clock; // @[FPMult.scala 18:33:@184.4]
  wire [23:0] multiplyMantissas_io_a; // @[FPMult.scala 18:33:@184.4]
  wire [23:0] multiplyMantissas_io_b; // @[FPMult.scala 18:33:@184.4]
  wire [47:0] multiplyMantissas_io_r; // @[FPMult.scala 18:33:@184.4]
  wire [22:0] zeroCheck_io_op1_m; // @[FPMult.scala 21:25:@191.4]
  wire [7:0] zeroCheck_io_op1_e; // @[FPMult.scala 21:25:@191.4]
  wire [22:0] zeroCheck_io_op2_m; // @[FPMult.scala 21:25:@191.4]
  wire [7:0] zeroCheck_io_op2_e; // @[FPMult.scala 21:25:@191.4]
  wire  zeroCheck_io_out; // @[FPMult.scala 21:25:@191.4]
  wire  addExponentsWithOffset_clock; // @[FPMult.scala 25:38:@198.4]
  wire [7:0] addExponentsWithOffset_io_e1; // @[FPMult.scala 25:38:@198.4]
  wire [7:0] addExponentsWithOffset_io_e2; // @[FPMult.scala 25:38:@198.4]
  wire [8:0] addExponentsWithOffset_io_eout; // @[FPMult.scala 25:38:@198.4]
  wire [8:0] addOne_io_e1; // @[FPMult.scala 31:22:@203.4]
  wire  addOne_io_add; // @[FPMult.scala 31:22:@203.4]
  wire [7:0] addOne_io_e_out; // @[FPMult.scala 31:22:@203.4]
  wire  addOne_io_underflow_out; // @[FPMult.scala 31:22:@203.4]
  reg [8:0] _T_25; // @[Reg.scala 11:16:@206.4]
  reg [31:0] _RAND_2;
  reg [8:0] _T_27; // @[Reg.scala 11:16:@210.4]
  reg [31:0] _RAND_3;
  wire [47:0] mantissaSelector_io_m_in; // @[FPMult.scala 34:32:@217.4]
  wire [22:0] mantissaSelector_io_m_out; // @[FPMult.scala 34:32:@217.4]
  wire  zeroHandler_clock; // @[FPMult.scala 37:27:@221.4]
  wire [22:0] zeroHandler_io_m_in; // @[FPMult.scala 37:27:@221.4]
  wire [7:0] zeroHandler_io_e_in; // @[FPMult.scala 37:27:@221.4]
  wire  zeroHandler_io_underflow_in; // @[FPMult.scala 37:27:@221.4]
  wire  zeroHandler_io_zero; // @[FPMult.scala 37:27:@221.4]
  wire [22:0] zeroHandler_io_out_m; // @[FPMult.scala 37:27:@221.4]
  wire [7:0] zeroHandler_io_out_e; // @[FPMult.scala 37:27:@221.4]
  reg  _T_31; // @[Reg.scala 11:16:@227.4]
  reg [31:0] _RAND_4;
  reg  _T_33; // @[Reg.scala 11:16:@231.4]
  reg [31:0] _RAND_5;
  wire [7:0] op1_e; // @[UFloat.scala 19:26:@175.4]
  wire [22:0] op1_m; // @[UFloat.scala 20:26:@177.4]
  wire [7:0] op2_e; // @[UFloat.scala 19:26:@180.4]
  wire [22:0] op2_m; // @[UFloat.scala 20:26:@182.4]
  wire [23:0] _T_20; // @[Cat.scala 30:58:@187.4]
  wire [23:0] _T_22; // @[Cat.scala 30:58:@189.4]
  wire [8:0] _GEN_0; // @[Reg.scala 12:19:@207.4]
  wire  _T_28; // @[FPMult.scala 33:47:@215.4]
  wire  _GEN_2; // @[Reg.scala 12:19:@228.4]
  wire [30:0] _T_34; // @[Cat.scala 30:58:@236.4]
  DSPMult multiplyMantissas ( // @[FPMult.scala 18:33:@184.4]
    .clock(multiplyMantissas_clock),
    .io_a(multiplyMantissas_io_a),
    .io_b(multiplyMantissas_io_b),
    .io_r(multiplyMantissas_io_r)
  );
  ZeroCheckComb zeroCheck ( // @[FPMult.scala 21:25:@191.4]
    .io_op1_m(zeroCheck_io_op1_m),
    .io_op1_e(zeroCheck_io_op1_e),
    .io_op2_m(zeroCheck_io_op2_m),
    .io_op2_e(zeroCheck_io_op2_e),
    .io_out(zeroCheck_io_out)
  );
  AddExponentsComb addExponentsWithOffset ( // @[FPMult.scala 25:38:@198.4]
    .clock(addExponentsWithOffset_clock),
    .io_e1(addExponentsWithOffset_io_e1),
    .io_e2(addExponentsWithOffset_io_e2),
    .io_eout(addExponentsWithOffset_io_eout)
  );
  AddOneToExpAndSubtractOffsetComb addOne ( // @[FPMult.scala 31:22:@203.4]
    .io_e1(addOne_io_e1),
    .io_add(addOne_io_add),
    .io_e_out(addOne_io_e_out),
    .io_underflow_out(addOne_io_underflow_out)
  );
  MantissaSelectorComb mantissaSelector ( // @[FPMult.scala 34:32:@217.4]
    .io_m_in(mantissaSelector_io_m_in),
    .io_m_out(mantissaSelector_io_m_out)
  );
  ZeroHandler zeroHandler ( // @[FPMult.scala 37:27:@221.4]
    .clock(zeroHandler_clock),
    .io_m_in(zeroHandler_io_m_in),
    .io_e_in(zeroHandler_io_e_in),
    .io_underflow_in(zeroHandler_io_underflow_in),
    .io_zero(zeroHandler_io_zero),
    .io_out_m(zeroHandler_io_out_m),
    .io_out_e(zeroHandler_io_out_e)
  );
  assign op1_e = op1_packed[30:23]; // @[UFloat.scala 19:26:@175.4]
  assign op1_m = op1_packed[22:0]; // @[UFloat.scala 20:26:@177.4]
  assign op2_e = op2_packed[30:23]; // @[UFloat.scala 19:26:@180.4]
  assign op2_m = op2_packed[22:0]; // @[UFloat.scala 20:26:@182.4]
  assign _T_20 = {1'h1,op1_m}; // @[Cat.scala 30:58:@187.4]
  assign _T_22 = {1'h1,op2_m}; // @[Cat.scala 30:58:@189.4]
  assign _GEN_0 = addExponentsWithOffset_io_eout; // @[Reg.scala 12:19:@207.4]
  assign _T_28 = multiplyMantissas_io_r[47:47]; // @[FPMult.scala 33:47:@215.4]
  assign _GEN_2 = zeroCheck_io_out; // @[Reg.scala 12:19:@228.4]
  assign _T_34 = {zeroHandler_io_out_e,zeroHandler_io_out_m}; // @[Cat.scala 30:58:@236.4]
  assign io_r = _T_34;
  assign multiplyMantissas_clock = clock;
  assign multiplyMantissas_io_a = _T_20;
  assign multiplyMantissas_io_b = _T_22;
  assign zeroCheck_io_op1_m = op1_m;
  assign zeroCheck_io_op1_e = op1_e;
  assign zeroCheck_io_op2_m = op2_m;
  assign zeroCheck_io_op2_e = op2_e;
  assign addExponentsWithOffset_clock = clock;
  assign addExponentsWithOffset_io_e1 = op1_e;
  assign addExponentsWithOffset_io_e2 = op2_e;
  assign addOne_io_e1 = _T_27;
  assign addOne_io_add = _T_28;
  assign mantissaSelector_io_m_in = multiplyMantissas_io_r;
  assign zeroHandler_clock = clock;
  assign zeroHandler_io_m_in = mantissaSelector_io_m_out;
  assign zeroHandler_io_e_in = addOne_io_e_out;
  assign zeroHandler_io_underflow_in = addOne_io_underflow_out;
  assign zeroHandler_io_zero = _T_33;
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
  _T_25 = _RAND_2[8:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_3 = {1{$random}};
  _T_27 = _RAND_3[8:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_4 = {1{$random}};
  _T_31 = _RAND_4[0:0];
  `endif // RANDOMIZE_REG_INIT
  `ifdef RANDOMIZE_REG_INIT
  _RAND_5 = {1{$random}};
  _T_33 = _RAND_5[0:0];
  `endif // RANDOMIZE_REG_INIT
  end
`endif // RANDOMIZE
  always @(posedge clock) begin
    op1_packed <= io_a;
    op2_packed <= io_b;
    _T_25 <= _GEN_0;
    _T_27 <= _T_25;
    _T_31 <= _GEN_2;
    _T_33 <= _T_31;
  end
endmodule
