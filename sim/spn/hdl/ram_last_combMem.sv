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

// VCS coverage exclude_file
module ram_last_combMem(	// src/main/scala/chisel3/util/Decoupled.scala:273:95
  input  R0_addr,
         R0_en,
         R0_clk,
         W0_addr,
         W0_en,
         W0_clk,
         W0_data,
  output R0_data);

  reg  Memory[0:1];	// src/main/scala/chisel3/util/Decoupled.scala:273:95
  wire _GEN;	// src/main/scala/chisel3/util/Decoupled.scala:273:95
  /* synopsys infer_mux_override */
  assign _GEN = Memory[R0_addr] /* cadence map_to_mux */;	// src/main/scala/chisel3/util/Decoupled.scala:273:95
  always @(posedge W0_clk) begin	// src/main/scala/chisel3/util/Decoupled.scala:273:95
    if (W0_en)	// src/main/scala/chisel3/util/Decoupled.scala:273:95
      Memory[W0_addr] <= W0_data;	// src/main/scala/chisel3/util/Decoupled.scala:273:95
  end // always @(posedge)
  `ifndef SYNTHESIS	// src/main/scala/chisel3/util/Decoupled.scala:273:95
    `ifdef RANDOMIZE_MEM_INIT	// src/main/scala/chisel3/util/Decoupled.scala:273:95
      integer initvar;	// src/main/scala/chisel3/util/Decoupled.scala:273:95
      reg [31:0] _RANDOM_MEM;	// src/main/scala/chisel3/util/Decoupled.scala:273:95
    `endif // RANDOMIZE_MEM_INIT
    `ifndef RANDOMIZE_REG_INIT	// src/main/scala/chisel3/util/Decoupled.scala:273:95
    `endif // not def RANDOMIZE_REG_INIT
    initial begin	// src/main/scala/chisel3/util/Decoupled.scala:273:95
      `INIT_RANDOM_PROLOG_	// src/main/scala/chisel3/util/Decoupled.scala:273:95
      `ifdef RANDOMIZE_MEM_INIT	// src/main/scala/chisel3/util/Decoupled.scala:273:95
        for (initvar = 0; initvar < 2; initvar = initvar + 1) begin
          _RANDOM_MEM = {`RANDOM};
          Memory[initvar] = _RANDOM_MEM[0:0];
        end	// src/main/scala/chisel3/util/Decoupled.scala:273:95
      `endif // RANDOMIZE_MEM_INIT
      `ifndef RANDOMIZE_REG_INIT	// src/main/scala/chisel3/util/Decoupled.scala:273:95
      `endif // not def RANDOMIZE_REG_INIT
    end // initial
  `endif // not def SYNTHESIS
  assign R0_data = R0_en ? _GEN : 1'bx;	// src/main/scala/chisel3/util/Decoupled.scala:273:95
endmodule


