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


module histogram_8(
  input         clk,
                rst,
  input  [7:0]  in_index,
  output [30:0] out_prob);

  wire [71:0][30:0] _GEN =
    {{31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3B864B8B},
     {31'h3C064B8A},
     {31'h3C064B8A},
     {31'h3CC97150},
     {31'h3DEB0432},
     {31'h3F53EF37}};
  reg  [30:0]       bufferReg;
  always @(posedge clk) begin
    if (rst)
      bufferReg <= 31'h0;
    else
      bufferReg <= _GEN[in_index[6:0]];
  end // always @(posedge)
  `ifndef SYNTHESIS
    `ifdef FIRRTL_BEFORE_INITIAL
      `FIRRTL_BEFORE_INITIAL
    `endif // FIRRTL_BEFORE_INITIAL
    initial begin
      automatic logic [31:0] _RANDOM_0;
      `ifdef INIT_RANDOM_PROLOG_
        `INIT_RANDOM_PROLOG_
      `endif // INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_REG_INIT
        _RANDOM_0 = `RANDOM;
        bufferReg = _RANDOM_0[30:0];
      `endif // RANDOMIZE_REG_INIT
    end // initial
    `ifdef FIRRTL_AFTER_INITIAL
      `FIRRTL_AFTER_INITIAL
    `endif // FIRRTL_AFTER_INITIAL
  `endif // not def SYNTHESIS
  assign out_prob = bufferReg;
endmodule

