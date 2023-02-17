// Generated by CIRCT unknown git version
// Standard header to adapt well known macros to our needs.
`ifdef RANDOMIZE_REG_INIT
  `define RANDOMIZE
`endif // RANDOMIZE_REG_INIT
`ifdef RANDOMIZE_MEM_INIT
  `define RANDOMIZE
`endif // RANDOMIZE_MEM_INIT

// RANDOM may be set to an expression that produces a 32-bit random unsigned value.
`ifndef RANDOM
  `define RANDOM $random
`endif // not def RANDOM

// Users can define INIT_RANDOM as general code that gets injected into the
// initializer block for modules with registers.
`ifndef INIT_RANDOM
  `define INIT_RANDOM
`endif // not def INIT_RANDOM

// If using random initialization, you can also define RANDOMIZE_DELAY to
// customize the delay used, otherwise 0.002 is used.
`ifndef RANDOMIZE_DELAY
  `define RANDOMIZE_DELAY 0.002
`endif // not def RANDOMIZE_DELAY

// Define INIT_RANDOM_PROLOG_ for use in our modules below.
`ifdef RANDOMIZE
  `ifdef VERILATOR
    `define INIT_RANDOM_PROLOG_ `INIT_RANDOM
  `else  // VERILATOR
    `define INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end
  `endif // VERILATOR
`else  // RANDOMIZE
  `define INIT_RANDOM_PROLOG_
`endif // RANDOMIZE


module spn_body(
  input         clk,
                rst,
  input  [7:0]  in_0,
                in_1,
                in_2,
                in_3,
                in_4,
  output [30:0] out_prob);

  wire [30:0] _instance_27_io_r;
  wire [30:0] _instance_26_io_r;
  wire [30:0] _instance_25_io_r;
  wire [30:0] _instance_24_io_r;
  wire [30:0] _instance_23_io_r;
  wire [30:0] _instance_22_io_r;
  wire [30:0] _instance_21_io_r;
  wire [30:0] _instance_20_out_prob;
  wire [30:0] _instance_19_out_prob;
  wire [30:0] _instance_18_out_prob;
  wire [30:0] _instance_17_out_prob;
  wire [30:0] _instance_16_out_prob;
  wire [30:0] _instance_15_io_r;
  wire [30:0] _instance_14_io_r;
  wire [30:0] _instance_13_io_r;
  wire [30:0] _instance_12_io_r;
  wire [30:0] _instance_11_out_prob;
  wire [30:0] _instance_10_out_prob;
  wire [30:0] _instance_9_out_prob;
  wire [30:0] _instance_8_out_prob;
  wire [30:0] _instance_7_out_prob;
  reg  [30:0] shiftReg;
  reg  [30:0] shiftReg_0;
  reg  [30:0] shiftReg_1;
  reg  [30:0] shiftReg_2;
  reg  [30:0] shiftReg_3;
  reg  [30:0] shiftReg_4;
  reg  [30:0] shiftReg_5;
  reg  [30:0] shiftReg_6;
  reg  [30:0] shiftReg_7;
  reg  [30:0] shiftReg_8;
  reg  [30:0] shiftReg_9;
  reg  [30:0] shiftReg_10;
  reg  [30:0] shiftReg_11;
  reg  [30:0] shiftReg_12;
  reg  [30:0] shiftReg_13;
  reg  [30:0] shiftReg_14;
  reg  [30:0] shiftReg_15;
  reg  [30:0] shiftReg_16;
  reg  [30:0] shiftReg_17;
  reg  [30:0] shiftReg_18;
  reg  [30:0] shiftReg_19;
  reg  [30:0] shiftReg_20;
  reg  [30:0] shiftReg_21;
  reg  [30:0] shiftReg_22;
  reg  [30:0] shiftReg_23;
  reg  [30:0] shiftReg_24;
  reg  [30:0] shiftReg_25;
  reg  [30:0] shiftReg_26;
  reg  [30:0] shiftReg_27;
  reg  [30:0] shiftReg_28;
  reg  [30:0] shiftReg_29;
  reg  [30:0] shiftReg_30;
  histogram_7 instance_7 (
    .clk      (clk),
    .rst      (rst),
    .in_index (in_0),
    .out_prob (_instance_7_out_prob)
  );
  histogram_8 instance_8 (
    .clk      (clk),
    .rst      (rst),
    .in_index (in_1),
    .out_prob (_instance_8_out_prob)
  );
  histogram_9 instance_9 (
    .clk      (clk),
    .rst      (rst),
    .in_index (in_2),
    .out_prob (_instance_9_out_prob)
  );
  reg  [30:0] shiftReg_31;
  reg  [30:0] shiftReg_32;
  reg  [30:0] shiftReg_33;
  reg  [30:0] shiftReg_34;
  reg  [30:0] shiftReg_35;
  histogram_10 instance_10 (
    .clk      (clk),
    .rst      (rst),
    .in_index (in_3),
    .out_prob (_instance_10_out_prob)
  );
  histogram_11 instance_11 (
    .clk      (clk),
    .rst      (rst),
    .in_index (in_4),
    .out_prob (_instance_11_out_prob)
  );
  FPMult instance_12 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_7_out_prob),
    .io_b  (_instance_8_out_prob),
    .io_r  (_instance_12_io_r)
  );
  FPMult instance_13 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_12_io_r),
    .io_b  (shiftReg_35),
    .io_r  (_instance_13_io_r)
  );
  FPMult instance_14 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_10_out_prob),
    .io_b  (_instance_11_out_prob),
    .io_r  (_instance_14_io_r)
  );
  reg  [30:0] shiftReg_36;
  reg  [30:0] shiftReg_37;
  reg  [30:0] shiftReg_38;
  reg  [30:0] shiftReg_39;
  reg  [30:0] shiftReg_40;
  FPMult instance_15 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_13_io_r),
    .io_b  (shiftReg_40),
    .io_r  (_instance_15_io_r)
  );
  histogram_16 instance_16 (
    .clk      (clk),
    .rst      (rst),
    .in_index (in_0),
    .out_prob (_instance_16_out_prob)
  );
  histogram_17 instance_17 (
    .clk      (clk),
    .rst      (rst),
    .in_index (in_1),
    .out_prob (_instance_17_out_prob)
  );
  histogram_18 instance_18 (
    .clk      (clk),
    .rst      (rst),
    .in_index (in_2),
    .out_prob (_instance_18_out_prob)
  );
  reg  [30:0] shiftReg_41;
  reg  [30:0] shiftReg_42;
  reg  [30:0] shiftReg_43;
  reg  [30:0] shiftReg_44;
  reg  [30:0] shiftReg_45;
  histogram_19 instance_19 (
    .clk      (clk),
    .rst      (rst),
    .in_index (in_3),
    .out_prob (_instance_19_out_prob)
  );
  histogram_20 instance_20 (
    .clk      (clk),
    .rst      (rst),
    .in_index (in_4),
    .out_prob (_instance_20_out_prob)
  );
  FPMult instance_21 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_16_out_prob),
    .io_b  (_instance_17_out_prob),
    .io_r  (_instance_21_io_r)
  );
  FPMult instance_22 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_21_io_r),
    .io_b  (shiftReg_45),
    .io_r  (_instance_22_io_r)
  );
  FPMult instance_23 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_19_out_prob),
    .io_b  (_instance_20_out_prob),
    .io_r  (_instance_23_io_r)
  );
  reg  [30:0] shiftReg_46;
  reg  [30:0] shiftReg_47;
  reg  [30:0] shiftReg_48;
  reg  [30:0] shiftReg_49;
  reg  [30:0] shiftReg_50;
  FPMult instance_24 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_22_io_r),
    .io_b  (shiftReg_50),
    .io_r  (_instance_24_io_r)
  );
  FPMult instance_25 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_15_io_r),
    .io_b  (shiftReg_30),
    .io_r  (_instance_25_io_r)
  );
  FPMult instance_26 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_24_io_r),
    .io_b  (shiftReg_14),
    .io_r  (_instance_26_io_r)
  );
  FPAdd instance_27 (
    .clock (clk),
    .reset (rst),
    .io_a  (_instance_25_io_r),
    .io_b  (_instance_26_io_r),
    .io_r  (_instance_27_io_r)
  );
  FPLog instance_28 (
    .clk   (clk),
    .rst   (rst),
    .in_a  (_instance_27_io_r),
    .out_b (out_prob)
  );
  always @(posedge clk) begin
    if (rst) begin
      shiftReg <= 31'h0;
      shiftReg_0 <= 31'h0;
      shiftReg_1 <= 31'h0;
      shiftReg_2 <= 31'h0;
      shiftReg_3 <= 31'h0;
      shiftReg_4 <= 31'h0;
      shiftReg_5 <= 31'h0;
      shiftReg_6 <= 31'h0;
      shiftReg_7 <= 31'h0;
      shiftReg_8 <= 31'h0;
      shiftReg_9 <= 31'h0;
      shiftReg_10 <= 31'h0;
      shiftReg_11 <= 31'h0;
      shiftReg_12 <= 31'h0;
      shiftReg_13 <= 31'h0;
      shiftReg_14 <= 31'h0;
      shiftReg_15 <= 31'h0;
      shiftReg_16 <= 31'h0;
      shiftReg_17 <= 31'h0;
      shiftReg_18 <= 31'h0;
      shiftReg_19 <= 31'h0;
      shiftReg_20 <= 31'h0;
      shiftReg_21 <= 31'h0;
      shiftReg_22 <= 31'h0;
      shiftReg_23 <= 31'h0;
      shiftReg_24 <= 31'h0;
      shiftReg_25 <= 31'h0;
      shiftReg_26 <= 31'h0;
      shiftReg_27 <= 31'h0;
      shiftReg_28 <= 31'h0;
      shiftReg_29 <= 31'h0;
      shiftReg_30 <= 31'h0;
      shiftReg_31 <= 31'h0;
      shiftReg_32 <= 31'h0;
      shiftReg_33 <= 31'h0;
      shiftReg_34 <= 31'h0;
      shiftReg_35 <= 31'h0;
      shiftReg_36 <= 31'h0;
      shiftReg_37 <= 31'h0;
      shiftReg_38 <= 31'h0;
      shiftReg_39 <= 31'h0;
      shiftReg_40 <= 31'h0;
      shiftReg_41 <= 31'h0;
      shiftReg_42 <= 31'h0;
      shiftReg_43 <= 31'h0;
      shiftReg_44 <= 31'h0;
      shiftReg_45 <= 31'h0;
      shiftReg_46 <= 31'h0;
      shiftReg_47 <= 31'h0;
      shiftReg_48 <= 31'h0;
      shiftReg_49 <= 31'h0;
      shiftReg_50 <= 31'h0;
    end
    else begin
      shiftReg <= 31'h3F4DDDDE;
      shiftReg_0 <= shiftReg;
      shiftReg_1 <= shiftReg_0;
      shiftReg_2 <= shiftReg_1;
      shiftReg_3 <= shiftReg_2;
      shiftReg_4 <= shiftReg_3;
      shiftReg_5 <= shiftReg_4;
      shiftReg_6 <= shiftReg_5;
      shiftReg_7 <= shiftReg_6;
      shiftReg_8 <= shiftReg_7;
      shiftReg_9 <= shiftReg_8;
      shiftReg_10 <= shiftReg_9;
      shiftReg_11 <= shiftReg_10;
      shiftReg_12 <= shiftReg_11;
      shiftReg_13 <= shiftReg_12;
      shiftReg_14 <= shiftReg_13;
      shiftReg_15 <= 31'h3E488889;
      shiftReg_16 <= shiftReg_15;
      shiftReg_17 <= shiftReg_16;
      shiftReg_18 <= shiftReg_17;
      shiftReg_19 <= shiftReg_18;
      shiftReg_20 <= shiftReg_19;
      shiftReg_21 <= shiftReg_20;
      shiftReg_22 <= shiftReg_21;
      shiftReg_23 <= shiftReg_22;
      shiftReg_24 <= shiftReg_23;
      shiftReg_25 <= shiftReg_24;
      shiftReg_26 <= shiftReg_25;
      shiftReg_27 <= shiftReg_26;
      shiftReg_28 <= shiftReg_27;
      shiftReg_29 <= shiftReg_28;
      shiftReg_30 <= shiftReg_29;
      shiftReg_31 <= _instance_9_out_prob;
      shiftReg_32 <= shiftReg_31;
      shiftReg_33 <= shiftReg_32;
      shiftReg_34 <= shiftReg_33;
      shiftReg_35 <= shiftReg_34;
      shiftReg_36 <= _instance_14_io_r;
      shiftReg_37 <= shiftReg_36;
      shiftReg_38 <= shiftReg_37;
      shiftReg_39 <= shiftReg_38;
      shiftReg_40 <= shiftReg_39;
      shiftReg_41 <= _instance_18_out_prob;
      shiftReg_42 <= shiftReg_41;
      shiftReg_43 <= shiftReg_42;
      shiftReg_44 <= shiftReg_43;
      shiftReg_45 <= shiftReg_44;
      shiftReg_46 <= _instance_23_io_r;
      shiftReg_47 <= shiftReg_46;
      shiftReg_48 <= shiftReg_47;
      shiftReg_49 <= shiftReg_48;
      shiftReg_50 <= shiftReg_49;
    end
  end // always @(posedge)
  `ifndef SYNTHESIS
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM_0;
      automatic logic [31:0] _RANDOM_1;
      automatic logic [31:0] _RANDOM_2;
      automatic logic [31:0] _RANDOM_3;
      automatic logic [31:0] _RANDOM_4;
      automatic logic [31:0] _RANDOM_5;
      automatic logic [31:0] _RANDOM_6;
      automatic logic [31:0] _RANDOM_7;
      automatic logic [31:0] _RANDOM_8;
      automatic logic [31:0] _RANDOM_9;
      automatic logic [31:0] _RANDOM_10;
      automatic logic [31:0] _RANDOM_11;
      automatic logic [31:0] _RANDOM_12;
      automatic logic [31:0] _RANDOM_13;
      automatic logic [31:0] _RANDOM_14;
      automatic logic [31:0] _RANDOM_15;
      automatic logic [31:0] _RANDOM_16;
      automatic logic [31:0] _RANDOM_17;
      automatic logic [31:0] _RANDOM_18;
      automatic logic [31:0] _RANDOM_19;
      automatic logic [31:0] _RANDOM_20;
      automatic logic [31:0] _RANDOM_21;
      automatic logic [31:0] _RANDOM_22;
      automatic logic [31:0] _RANDOM_23;
      automatic logic [31:0] _RANDOM_24;
      automatic logic [31:0] _RANDOM_25;
      automatic logic [31:0] _RANDOM_26;
      automatic logic [31:0] _RANDOM_27;
      automatic logic [31:0] _RANDOM_28;
      automatic logic [31:0] _RANDOM_29;
      automatic logic [31:0] _RANDOM_30;
      automatic logic [31:0] _RANDOM_31;
      automatic logic [31:0] _RANDOM_32;
      automatic logic [31:0] _RANDOM_33;
      automatic logic [31:0] _RANDOM_34;
      automatic logic [31:0] _RANDOM_35;
      automatic logic [31:0] _RANDOM_36;
      automatic logic [31:0] _RANDOM_37;
      automatic logic [31:0] _RANDOM_38;
      automatic logic [31:0] _RANDOM_39;
      automatic logic [31:0] _RANDOM_40;
      automatic logic [31:0] _RANDOM_41;
      automatic logic [31:0] _RANDOM_42;
      automatic logic [31:0] _RANDOM_43;
      automatic logic [31:0] _RANDOM_44;
      automatic logic [31:0] _RANDOM_45;
      automatic logic [31:0] _RANDOM_46;
      automatic logic [31:0] _RANDOM_47;
      automatic logic [31:0] _RANDOM_48;
      automatic logic [31:0] _RANDOM_49;
      automatic logic [31:0] _RANDOM_50;
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        _RANDOM_0 = `RANDOM;
        _RANDOM_1 = `RANDOM;
        _RANDOM_2 = `RANDOM;
        _RANDOM_3 = `RANDOM;
        _RANDOM_4 = `RANDOM;
        _RANDOM_5 = `RANDOM;
        _RANDOM_6 = `RANDOM;
        _RANDOM_7 = `RANDOM;
        _RANDOM_8 = `RANDOM;
        _RANDOM_9 = `RANDOM;
        _RANDOM_10 = `RANDOM;
        _RANDOM_11 = `RANDOM;
        _RANDOM_12 = `RANDOM;
        _RANDOM_13 = `RANDOM;
        _RANDOM_14 = `RANDOM;
        _RANDOM_15 = `RANDOM;
        _RANDOM_16 = `RANDOM;
        _RANDOM_17 = `RANDOM;
        _RANDOM_18 = `RANDOM;
        _RANDOM_19 = `RANDOM;
        _RANDOM_20 = `RANDOM;
        _RANDOM_21 = `RANDOM;
        _RANDOM_22 = `RANDOM;
        _RANDOM_23 = `RANDOM;
        _RANDOM_24 = `RANDOM;
        _RANDOM_25 = `RANDOM;
        _RANDOM_26 = `RANDOM;
        _RANDOM_27 = `RANDOM;
        _RANDOM_28 = `RANDOM;
        _RANDOM_29 = `RANDOM;
        _RANDOM_30 = `RANDOM;
        _RANDOM_31 = `RANDOM;
        _RANDOM_32 = `RANDOM;
        _RANDOM_33 = `RANDOM;
        _RANDOM_34 = `RANDOM;
        _RANDOM_35 = `RANDOM;
        _RANDOM_36 = `RANDOM;
        _RANDOM_37 = `RANDOM;
        _RANDOM_38 = `RANDOM;
        _RANDOM_39 = `RANDOM;
        _RANDOM_40 = `RANDOM;
        _RANDOM_41 = `RANDOM;
        _RANDOM_42 = `RANDOM;
        _RANDOM_43 = `RANDOM;
        _RANDOM_44 = `RANDOM;
        _RANDOM_45 = `RANDOM;
        _RANDOM_46 = `RANDOM;
        _RANDOM_47 = `RANDOM;
        _RANDOM_48 = `RANDOM;
        _RANDOM_49 = `RANDOM;
        _RANDOM_50 = `RANDOM;
        shiftReg = _RANDOM_0[30:0];
        shiftReg_0 = {_RANDOM_0[31], _RANDOM_1[29:0]};
        shiftReg_1 = {_RANDOM_1[31:30], _RANDOM_2[28:0]};
        shiftReg_2 = {_RANDOM_2[31:29], _RANDOM_3[27:0]};
        shiftReg_3 = {_RANDOM_3[31:28], _RANDOM_4[26:0]};
        shiftReg_4 = {_RANDOM_4[31:27], _RANDOM_5[25:0]};
        shiftReg_5 = {_RANDOM_5[31:26], _RANDOM_6[24:0]};
        shiftReg_6 = {_RANDOM_6[31:25], _RANDOM_7[23:0]};
        shiftReg_7 = {_RANDOM_7[31:24], _RANDOM_8[22:0]};
        shiftReg_8 = {_RANDOM_8[31:23], _RANDOM_9[21:0]};
        shiftReg_9 = {_RANDOM_9[31:22], _RANDOM_10[20:0]};
        shiftReg_10 = {_RANDOM_10[31:21], _RANDOM_11[19:0]};
        shiftReg_11 = {_RANDOM_11[31:20], _RANDOM_12[18:0]};
        shiftReg_12 = {_RANDOM_12[31:19], _RANDOM_13[17:0]};
        shiftReg_13 = {_RANDOM_13[31:18], _RANDOM_14[16:0]};
        shiftReg_14 = {_RANDOM_14[31:17], _RANDOM_15[15:0]};
        shiftReg_15 = {_RANDOM_15[31:16], _RANDOM_16[14:0]};
        shiftReg_16 = {_RANDOM_16[31:15], _RANDOM_17[13:0]};
        shiftReg_17 = {_RANDOM_17[31:14], _RANDOM_18[12:0]};
        shiftReg_18 = {_RANDOM_18[31:13], _RANDOM_19[11:0]};
        shiftReg_19 = {_RANDOM_19[31:12], _RANDOM_20[10:0]};
        shiftReg_20 = {_RANDOM_20[31:11], _RANDOM_21[9:0]};
        shiftReg_21 = {_RANDOM_21[31:10], _RANDOM_22[8:0]};
        shiftReg_22 = {_RANDOM_22[31:9], _RANDOM_23[7:0]};
        shiftReg_23 = {_RANDOM_23[31:8], _RANDOM_24[6:0]};
        shiftReg_24 = {_RANDOM_24[31:7], _RANDOM_25[5:0]};
        shiftReg_25 = {_RANDOM_25[31:6], _RANDOM_26[4:0]};
        shiftReg_26 = {_RANDOM_26[31:5], _RANDOM_27[3:0]};
        shiftReg_27 = {_RANDOM_27[31:4], _RANDOM_28[2:0]};
        shiftReg_28 = {_RANDOM_28[31:3], _RANDOM_29[1:0]};
        shiftReg_29 = {_RANDOM_29[31:2], _RANDOM_30[0]};
        shiftReg_30 = _RANDOM_30[31:1];
        shiftReg_31 = _RANDOM_31[30:0];
        shiftReg_32 = {_RANDOM_31[31], _RANDOM_32[29:0]};
        shiftReg_33 = {_RANDOM_32[31:30], _RANDOM_33[28:0]};
        shiftReg_34 = {_RANDOM_33[31:29], _RANDOM_34[27:0]};
        shiftReg_35 = {_RANDOM_34[31:28], _RANDOM_35[26:0]};
        shiftReg_36 = {_RANDOM_35[31:27], _RANDOM_36[25:0]};
        shiftReg_37 = {_RANDOM_36[31:26], _RANDOM_37[24:0]};
        shiftReg_38 = {_RANDOM_37[31:25], _RANDOM_38[23:0]};
        shiftReg_39 = {_RANDOM_38[31:24], _RANDOM_39[22:0]};
        shiftReg_40 = {_RANDOM_39[31:23], _RANDOM_40[21:0]};
        shiftReg_41 = {_RANDOM_40[31:22], _RANDOM_41[20:0]};
        shiftReg_42 = {_RANDOM_41[31:21], _RANDOM_42[19:0]};
        shiftReg_43 = {_RANDOM_42[31:20], _RANDOM_43[18:0]};
        shiftReg_44 = {_RANDOM_43[31:19], _RANDOM_44[17:0]};
        shiftReg_45 = {_RANDOM_44[31:18], _RANDOM_45[16:0]};
        shiftReg_46 = {_RANDOM_45[31:17], _RANDOM_46[15:0]};
        shiftReg_47 = {_RANDOM_46[31:16], _RANDOM_47[14:0]};
        shiftReg_48 = {_RANDOM_47[31:15], _RANDOM_48[13:0]};
        shiftReg_49 = {_RANDOM_48[31:14], _RANDOM_49[12:0]};
        shiftReg_50 = {_RANDOM_49[31:13], _RANDOM_50[11:0]};
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // not def SYNTHESIS
endmodule
