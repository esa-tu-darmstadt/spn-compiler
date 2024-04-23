module Queue_0(
  input         clock,
  input         reset,
  output        io_enq_ready, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  input         io_enq_valid, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  input  [31:0] io_enq_bits, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  input         io_deq_ready, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  output        io_deq_valid, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  output [31:0] io_deq_bits // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
);
`ifdef RANDOMIZE_MEM_INIT
  reg [31:0] _RAND_0;
`endif // RANDOMIZE_MEM_INIT
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
  reg [31:0] _RAND_3;
`endif // RANDOMIZE_REG_INIT
  reg [31:0] ram [0:1]; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire  ram_io_deq_bits_MPORT_en; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire  ram_io_deq_bits_MPORT_addr; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire [31:0] ram_io_deq_bits_MPORT_data; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire [31:0] ram_MPORT_data; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire  ram_MPORT_addr; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire  ram_MPORT_mask; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire  ram_MPORT_en; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  reg  enq_ptr_value; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
  reg  deq_ptr_value; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
  reg  maybe_full; // @[src/main/scala/chisel3/util/Decoupled.scala 277:27]
  wire  ptr_match = enq_ptr_value == deq_ptr_value; // @[src/main/scala/chisel3/util/Decoupled.scala 278:33]
  wire  empty = ptr_match & ~maybe_full; // @[src/main/scala/chisel3/util/Decoupled.scala 279:25]
  wire  full = ptr_match & maybe_full; // @[src/main/scala/chisel3/util/Decoupled.scala 280:24]
  wire  do_enq = io_enq_ready & io_enq_valid; // @[src/main/scala/chisel3/util/Decoupled.scala 52:35]
  wire  do_deq = io_deq_ready & io_deq_valid; // @[src/main/scala/chisel3/util/Decoupled.scala 52:35]
  assign ram_io_deq_bits_MPORT_en = 1'h1;
  assign ram_io_deq_bits_MPORT_addr = deq_ptr_value;
  assign ram_io_deq_bits_MPORT_data = ram[ram_io_deq_bits_MPORT_addr]; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  assign ram_MPORT_data = io_enq_bits;
  assign ram_MPORT_addr = enq_ptr_value;
  assign ram_MPORT_mask = 1'h1;
  assign ram_MPORT_en = io_enq_ready & io_enq_valid;
  assign io_enq_ready = ~full; // @[src/main/scala/chisel3/util/Decoupled.scala 304:19]
  assign io_deq_valid = ~empty; // @[src/main/scala/chisel3/util/Decoupled.scala 303:19]
  assign io_deq_bits = ram_io_deq_bits_MPORT_data; // @[src/main/scala/chisel3/util/Decoupled.scala 311:17]
  always @(posedge clock) begin
    if (ram_MPORT_en & ram_MPORT_mask) begin
      ram[ram_MPORT_addr] <= ram_MPORT_data; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
    end
    if (reset) begin // @[src/main/scala/chisel3/util/Counter.scala 61:40]
      enq_ptr_value <= 1'h0; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
    end else if (do_enq) begin // @[src/main/scala/chisel3/util/Decoupled.scala 287:16]
      enq_ptr_value <= enq_ptr_value + 1'h1; // @[src/main/scala/chisel3/util/Counter.scala 77:15]
    end
    if (reset) begin // @[src/main/scala/chisel3/util/Counter.scala 61:40]
      deq_ptr_value <= 1'h0; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
    end else if (do_deq) begin // @[src/main/scala/chisel3/util/Decoupled.scala 291:16]
      deq_ptr_value <= deq_ptr_value + 1'h1; // @[src/main/scala/chisel3/util/Counter.scala 77:15]
    end
    if (reset) begin // @[src/main/scala/chisel3/util/Decoupled.scala 277:27]
      maybe_full <= 1'h0; // @[src/main/scala/chisel3/util/Decoupled.scala 277:27]
    end else if (do_enq != do_deq) begin // @[src/main/scala/chisel3/util/Decoupled.scala 294:27]
      maybe_full <= do_enq; // @[src/main/scala/chisel3/util/Decoupled.scala 295:16]
    end
  end
// Register and memory initialization
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
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_MEM_INIT
  _RAND_0 = {1{`RANDOM}};
  for (initvar = 0; initvar < 2; initvar = initvar+1)
    ram[initvar] = _RAND_0[31:0];
`endif // RANDOMIZE_MEM_INIT
`ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{`RANDOM}};
  enq_ptr_value = _RAND_1[0:0];
  _RAND_2 = {1{`RANDOM}};
  deq_ptr_value = _RAND_2[0:0];
  _RAND_3 = {1{`RANDOM}};
  maybe_full = _RAND_3[0:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
module Queue_3(
  input         clock,
  input         reset,
  output        io_enq_ready, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  input         io_enq_valid, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  input  [31:0] io_enq_bits_data, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  input         io_deq_ready, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  output        io_deq_valid, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  output [31:0] io_deq_bits_data // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
);
`ifdef RANDOMIZE_MEM_INIT
  reg [31:0] _RAND_0;
`endif // RANDOMIZE_MEM_INIT
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
  reg [31:0] _RAND_3;
`endif // RANDOMIZE_REG_INIT
  reg [31:0] ram_data [0:1]; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire  ram_data_io_deq_bits_MPORT_en; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire  ram_data_io_deq_bits_MPORT_addr; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire [31:0] ram_data_io_deq_bits_MPORT_data; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire [31:0] ram_data_MPORT_data; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire  ram_data_MPORT_addr; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire  ram_data_MPORT_mask; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  wire  ram_data_MPORT_en; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  reg  enq_ptr_value; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
  reg  deq_ptr_value; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
  reg  maybe_full; // @[src/main/scala/chisel3/util/Decoupled.scala 277:27]
  wire  ptr_match = enq_ptr_value == deq_ptr_value; // @[src/main/scala/chisel3/util/Decoupled.scala 278:33]
  wire  empty = ptr_match & ~maybe_full; // @[src/main/scala/chisel3/util/Decoupled.scala 279:25]
  wire  full = ptr_match & maybe_full; // @[src/main/scala/chisel3/util/Decoupled.scala 280:24]
  wire  do_enq = io_enq_ready & io_enq_valid; // @[src/main/scala/chisel3/util/Decoupled.scala 52:35]
  wire  do_deq = io_deq_ready & io_deq_valid; // @[src/main/scala/chisel3/util/Decoupled.scala 52:35]
  assign ram_data_io_deq_bits_MPORT_en = 1'h1;
  assign ram_data_io_deq_bits_MPORT_addr = deq_ptr_value;
  assign ram_data_io_deq_bits_MPORT_data = ram_data[ram_data_io_deq_bits_MPORT_addr]; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
  assign ram_data_MPORT_data = io_enq_bits_data;
  assign ram_data_MPORT_addr = enq_ptr_value;
  assign ram_data_MPORT_mask = 1'h1;
  assign ram_data_MPORT_en = io_enq_ready & io_enq_valid;
  assign io_enq_ready = ~full; // @[src/main/scala/chisel3/util/Decoupled.scala 304:19]
  assign io_deq_valid = ~empty; // @[src/main/scala/chisel3/util/Decoupled.scala 303:19]
  assign io_deq_bits_data = ram_data_io_deq_bits_MPORT_data; // @[src/main/scala/chisel3/util/Decoupled.scala 311:17]
  always @(posedge clock) begin
    if (ram_data_MPORT_en & ram_data_MPORT_mask) begin
      ram_data[ram_data_MPORT_addr] <= ram_data_MPORT_data; // @[src/main/scala/chisel3/util/Decoupled.scala 274:95]
    end
    if (reset) begin // @[src/main/scala/chisel3/util/Counter.scala 61:40]
      enq_ptr_value <= 1'h0; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
    end else if (do_enq) begin // @[src/main/scala/chisel3/util/Decoupled.scala 287:16]
      enq_ptr_value <= enq_ptr_value + 1'h1; // @[src/main/scala/chisel3/util/Counter.scala 77:15]
    end
    if (reset) begin // @[src/main/scala/chisel3/util/Counter.scala 61:40]
      deq_ptr_value <= 1'h0; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
    end else if (do_deq) begin // @[src/main/scala/chisel3/util/Decoupled.scala 291:16]
      deq_ptr_value <= deq_ptr_value + 1'h1; // @[src/main/scala/chisel3/util/Counter.scala 77:15]
    end
    if (reset) begin // @[src/main/scala/chisel3/util/Decoupled.scala 277:27]
      maybe_full <= 1'h0; // @[src/main/scala/chisel3/util/Decoupled.scala 277:27]
    end else if (do_enq != do_deq) begin // @[src/main/scala/chisel3/util/Decoupled.scala 294:27]
      maybe_full <= do_enq; // @[src/main/scala/chisel3/util/Decoupled.scala 295:16]
    end
  end
// Register and memory initialization
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
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_MEM_INIT
  _RAND_0 = {1{`RANDOM}};
  for (initvar = 0; initvar < 2; initvar = initvar+1)
    ram_data[initvar] = _RAND_0[31:0];
`endif // RANDOMIZE_MEM_INIT
`ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{`RANDOM}};
  enq_ptr_value = _RAND_1[0:0];
  _RAND_2 = {1{`RANDOM}};
  deq_ptr_value = _RAND_2[0:0];
  _RAND_3 = {1{`RANDOM}};
  maybe_full = _RAND_3[0:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
module Queue_4(
  input   clock,
  input   reset,
  output  io_enq_ready, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  input   io_enq_valid, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  input   io_deq_ready, // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
  output  io_deq_valid // @[src/main/scala/chisel3/util/Decoupled.scala 273:14]
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
`endif // RANDOMIZE_REG_INIT
  reg  enq_ptr_value; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
  reg  deq_ptr_value; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
  reg  maybe_full; // @[src/main/scala/chisel3/util/Decoupled.scala 277:27]
  wire  ptr_match = enq_ptr_value == deq_ptr_value; // @[src/main/scala/chisel3/util/Decoupled.scala 278:33]
  wire  empty = ptr_match & ~maybe_full; // @[src/main/scala/chisel3/util/Decoupled.scala 279:25]
  wire  full = ptr_match & maybe_full; // @[src/main/scala/chisel3/util/Decoupled.scala 280:24]
  wire  do_enq = io_enq_ready & io_enq_valid; // @[src/main/scala/chisel3/util/Decoupled.scala 52:35]
  wire  do_deq = io_deq_ready & io_deq_valid; // @[src/main/scala/chisel3/util/Decoupled.scala 52:35]
  assign io_enq_ready = ~full; // @[src/main/scala/chisel3/util/Decoupled.scala 304:19]
  assign io_deq_valid = ~empty; // @[src/main/scala/chisel3/util/Decoupled.scala 303:19]
  always @(posedge clock) begin
    if (reset) begin // @[src/main/scala/chisel3/util/Counter.scala 61:40]
      enq_ptr_value <= 1'h0; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
    end else if (do_enq) begin // @[src/main/scala/chisel3/util/Decoupled.scala 287:16]
      enq_ptr_value <= enq_ptr_value + 1'h1; // @[src/main/scala/chisel3/util/Counter.scala 77:15]
    end
    if (reset) begin // @[src/main/scala/chisel3/util/Counter.scala 61:40]
      deq_ptr_value <= 1'h0; // @[src/main/scala/chisel3/util/Counter.scala 61:40]
    end else if (do_deq) begin // @[src/main/scala/chisel3/util/Decoupled.scala 291:16]
      deq_ptr_value <= deq_ptr_value + 1'h1; // @[src/main/scala/chisel3/util/Counter.scala 77:15]
    end
    if (reset) begin // @[src/main/scala/chisel3/util/Decoupled.scala 277:27]
      maybe_full <= 1'h0; // @[src/main/scala/chisel3/util/Decoupled.scala 277:27]
    end else if (do_enq != do_deq) begin // @[src/main/scala/chisel3/util/Decoupled.scala 294:27]
      maybe_full <= do_enq; // @[src/main/scala/chisel3/util/Decoupled.scala 295:16]
    end
  end
// Register and memory initialization
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
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  enq_ptr_value = _RAND_0[0:0];
  _RAND_1 = {1{`RANDOM}};
  deq_ptr_value = _RAND_1[0:0];
  _RAND_2 = {1{`RANDOM}};
  maybe_full = _RAND_2[0:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
module DummyWrapper(
  input          clock,
  input          reset,
  input          S_AXI_LITE_awvalid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  output         S_AXI_LITE_awready, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  input  [31:0]  S_AXI_LITE_awaddr, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  input  [2:0]   S_AXI_LITE_awprot, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  input          S_AXI_LITE_wvalid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  output         S_AXI_LITE_wready, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  input  [31:0]  S_AXI_LITE_wdata, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  input  [3:0]   S_AXI_LITE_wstrb, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  output         S_AXI_LITE_bvalid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  input          S_AXI_LITE_bready, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  output [1:0]   S_AXI_LITE_bresp, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  input          S_AXI_LITE_arvalid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  output         S_AXI_LITE_arready, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  input  [31:0]  S_AXI_LITE_araddr, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  input  [2:0]   S_AXI_LITE_arprot, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  output         S_AXI_LITE_rvalid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  input          S_AXI_LITE_rready, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  output [31:0]  S_AXI_LITE_rdata, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  output [1:0]   S_AXI_LITE_rresp, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 13:22]
  output         M_AXI_awvalid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input          M_AXI_awready, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output         M_AXI_awid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [31:0]  M_AXI_awaddr, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [7:0]   M_AXI_awlen, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [2:0]   M_AXI_awsize, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [1:0]   M_AXI_awburst, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output         M_AXI_awlock, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [3:0]   M_AXI_awcache, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [2:0]   M_AXI_awprot, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [3:0]   M_AXI_awqos, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [3:0]   M_AXI_awregion, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output         M_AXI_wvalid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input          M_AXI_wready, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [511:0] M_AXI_wdata, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [63:0]  M_AXI_wstrb, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output         M_AXI_wlast, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input          M_AXI_bvalid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output         M_AXI_bready, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input          M_AXI_bid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input  [1:0]   M_AXI_bresp, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output         M_AXI_arvalid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input          M_AXI_arready, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output         M_AXI_arid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [31:0]  M_AXI_araddr, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [7:0]   M_AXI_arlen, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [2:0]   M_AXI_arsize, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [1:0]   M_AXI_arburst, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output         M_AXI_arlock, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [3:0]   M_AXI_arcache, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [2:0]   M_AXI_arprot, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [3:0]   M_AXI_arqos, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output [3:0]   M_AXI_arregion, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input          M_AXI_rvalid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output         M_AXI_rready, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input          M_AXI_rid, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input  [511:0] M_AXI_rdata, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input  [1:0]   M_AXI_rresp, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  input          M_AXI_rlast, // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 14:17]
  output         interrupt // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 15:21]
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
  reg [31:0] _RAND_3;
  reg [31:0] _RAND_4;
  reg [31:0] _RAND_5;
  reg [31:0] _RAND_6;
  reg [31:0] _RAND_7;
  reg [31:0] _RAND_8;
  reg [31:0] _RAND_9;
  reg [31:0] _RAND_10;
  reg [31:0] _RAND_11;
  reg [31:0] _RAND_12;
  reg [31:0] _RAND_13;
  reg [31:0] _RAND_14;
  reg [31:0] _RAND_15;
  reg [31:0] _RAND_16;
  reg [31:0] _RAND_17;
  reg [31:0] _RAND_18;
  reg [31:0] _RAND_19;
  reg [31:0] _RAND_20;
  reg [31:0] _RAND_21;
  reg [31:0] _RAND_22;
  reg [31:0] _RAND_23;
  reg [31:0] _RAND_24;
  reg [31:0] _RAND_25;
  reg [31:0] _RAND_26;
  reg [31:0] _RAND_27;
  reg [31:0] _RAND_28;
  reg [31:0] _RAND_29;
  reg [31:0] _RAND_30;
  reg [31:0] _RAND_31;
  reg [31:0] _RAND_32;
  reg [31:0] _RAND_33;
`endif // RANDOMIZE_REG_INIT
  wire  rAddressQo_Queue_0_clock; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rAddressQo_Queue_0_reset; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rAddressQo_Queue_0_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rAddressQo_Queue_0_io_enq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire [31:0] rAddressQo_Queue_0_io_enq_bits; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rAddressQo_Queue_0_io_deq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rAddressQo_Queue_0_io_deq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire [31:0] rAddressQo_Queue_0_io_deq_bits; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rDataQo_Queue_1_clock; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rDataQo_Queue_1_reset; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rDataQo_Queue_1_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rDataQo_Queue_1_io_enq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire [31:0] rDataQo_Queue_1_io_enq_bits; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rDataQo_Queue_1_io_deq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  rDataQo_Queue_1_io_deq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire [31:0] rDataQo_Queue_1_io_deq_bits; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wAddrQo_Queue_2_clock; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wAddrQo_Queue_2_reset; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wAddrQo_Queue_2_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wAddrQo_Queue_2_io_enq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire [31:0] wAddrQo_Queue_2_io_enq_bits; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wAddrQo_Queue_2_io_deq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wAddrQo_Queue_2_io_deq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire [31:0] wAddrQo_Queue_2_io_deq_bits; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wDataQo_Queue_3_clock; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wDataQo_Queue_3_reset; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wDataQo_Queue_3_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wDataQo_Queue_3_io_enq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire [31:0] wDataQo_Queue_3_io_enq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wDataQo_Queue_3_io_deq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  wDataQo_Queue_3_io_deq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire [31:0] wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validAQ_Queue_4_clock; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validAQ_Queue_4_reset; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validAQ_Queue_4_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validAQ_Queue_4_io_enq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validAQ_Queue_4_io_deq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validAQ_Queue_4_io_deq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validDQ_Queue_5_clock; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validDQ_Queue_5_reset; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validDQ_Queue_5_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validDQ_Queue_5_io_enq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validDQ_Queue_5_io_deq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  wire  validDQ_Queue_5_io_deq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
  reg [31:0] registers_0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_1; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_2; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_3; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_4; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_5; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_6; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_7; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_8; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_9; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_10; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_11; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_12; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_13; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_14; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_15; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_16; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_17; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_18; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_19; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_20; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_21; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_22; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_23; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_24; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_25; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_26; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  reg [31:0] registers_27; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
  wire  rAddressQi_ready = rAddressQo_Queue_0_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 46:17 src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 59:26]
  wire  rAddressQi_valid = rAddressQi_ready & S_AXI_LITE_arvalid; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 83:28]
  reg  dState; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 88:25]
  reg  dValidBuf; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 89:28]
  reg [31:0] dataBuf; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 90:26]
  wire  _GEN_2 = rDataQo_Queue_1_io_deq_valid | dValidBuf; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 100:19 89:28 99:28]
  wire  _GEN_3 = rDataQo_Queue_1_io_deq_valid; // @[src/main/scala/chisel3/util/Decoupled.scala 83:20 90:20 src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 99:28]
  wire  _GEN_5 = rDataQo_Queue_1_io_deq_valid | dState; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 102:16 88:25 99:28]
  wire  rDataQi_ready = rDataQo_Queue_1_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 46:17 src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 62:23]
  wire  rDataQi_valid = rAddressQo_Queue_0_io_deq_valid & rDataQi_ready; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 72:27]
  wire [31:0] _readIdx_T = rAddressQo_Queue_0_io_deq_bits & 32'hff; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 74:23]
  wire [31:0] readIdx = rDataQi_valid ? {{2'd0}, _readIdx_T[31:2]} : 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 72:45 74:15 70:27]
  wire [31:0] _GEN_17 = 5'h1 == readIdx[4:0] ? registers_1 : registers_0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_18 = 5'h2 == readIdx[4:0] ? registers_2 : _GEN_17; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_19 = 5'h3 == readIdx[4:0] ? registers_3 : _GEN_18; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_20 = 5'h4 == readIdx[4:0] ? registers_4 : _GEN_19; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_21 = 5'h5 == readIdx[4:0] ? registers_5 : _GEN_20; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_22 = 5'h6 == readIdx[4:0] ? registers_6 : _GEN_21; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_23 = 5'h7 == readIdx[4:0] ? registers_7 : _GEN_22; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_24 = 5'h8 == readIdx[4:0] ? registers_8 : _GEN_23; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_25 = 5'h9 == readIdx[4:0] ? registers_9 : _GEN_24; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_26 = 5'ha == readIdx[4:0] ? registers_10 : _GEN_25; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_27 = 5'hb == readIdx[4:0] ? registers_11 : _GEN_26; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_28 = 5'hc == readIdx[4:0] ? registers_12 : _GEN_27; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_29 = 5'hd == readIdx[4:0] ? registers_13 : _GEN_28; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_30 = 5'he == readIdx[4:0] ? registers_14 : _GEN_29; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_31 = 5'hf == readIdx[4:0] ? registers_15 : _GEN_30; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_32 = 5'h10 == readIdx[4:0] ? registers_16 : _GEN_31; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_33 = 5'h11 == readIdx[4:0] ? registers_17 : _GEN_32; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_34 = 5'h12 == readIdx[4:0] ? registers_18 : _GEN_33; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_35 = 5'h13 == readIdx[4:0] ? registers_19 : _GEN_34; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_36 = 5'h14 == readIdx[4:0] ? registers_20 : _GEN_35; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_37 = 5'h15 == readIdx[4:0] ? registers_21 : _GEN_36; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_38 = 5'h16 == readIdx[4:0] ? registers_22 : _GEN_37; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_39 = 5'h17 == readIdx[4:0] ? registers_23 : _GEN_38; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_40 = 5'h18 == readIdx[4:0] ? registers_24 : _GEN_39; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_41 = 5'h19 == readIdx[4:0] ? registers_25 : _GEN_40; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_42 = 5'h1a == readIdx[4:0] ? registers_26 : _GEN_41; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  wire [31:0] _GEN_43 = 5'h1b == readIdx[4:0] ? registers_27 : _GEN_42; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 75:{11,11}]
  reg  rState; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 23:25]
  reg  rValidBuf; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 24:28]
  wire  _T_5 = validAQ_Queue_4_io_deq_valid & validDQ_Queue_5_io_deq_valid; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 32:26]
  wire  _GEN_48 = validAQ_Queue_4_io_deq_valid & validDQ_Queue_5_io_deq_valid | rValidBuf; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 32:44 33:19 24:28]
  wire  _GEN_50 = validAQ_Queue_4_io_deq_valid & validDQ_Queue_5_io_deq_valid | rState; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 32:44 36:16 23:25]
  wire  wAddrQi_ready = wAddrQo_Queue_2_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 46:17 src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 79:23]
  wire  validAddress_ready = validAQ_Queue_4_io_enq_ready; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 17:28 src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 46:17]
  wire  wAddrQi_valid = wAddrQi_ready & S_AXI_LITE_awvalid; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 51:29]
  wire  wDataQi_ready = wDataQo_Queue_3_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 46:17 src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 82:23]
  wire  validData_ready = validDQ_Queue_5_io_enq_ready; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 18:25 src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 46:17]
  wire  wDataQi_valid = wDataQi_ready & S_AXI_LITE_wvalid; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 63:26]
  wire [31:0] _writeIdx_T = wAddrQo_Queue_2_io_deq_bits & 32'hff; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 93:34]
  wire [29:0] _GEN_92 = wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid ? _writeIdx_T[31:2] : 30'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42 93:16 89:28]
  wire [7:0] writeIdx = _GEN_92[7:0]; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 89:28]
  reg [31:0] counter; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 23:24]
  wire [31:0] _counter_T_1 = counter - 32'h1; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 30:24]
  Queue_0 rAddressQo_Queue_0 ( // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
    .clock(rAddressQo_Queue_0_clock),
    .reset(rAddressQo_Queue_0_reset),
    .io_enq_ready(rAddressQo_Queue_0_io_enq_ready),
    .io_enq_valid(rAddressQo_Queue_0_io_enq_valid),
    .io_enq_bits(rAddressQo_Queue_0_io_enq_bits),
    .io_deq_ready(rAddressQo_Queue_0_io_deq_ready),
    .io_deq_valid(rAddressQo_Queue_0_io_deq_valid),
    .io_deq_bits(rAddressQo_Queue_0_io_deq_bits)
  );
  Queue_0 rDataQo_Queue_1 ( // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
    .clock(rDataQo_Queue_1_clock),
    .reset(rDataQo_Queue_1_reset),
    .io_enq_ready(rDataQo_Queue_1_io_enq_ready),
    .io_enq_valid(rDataQo_Queue_1_io_enq_valid),
    .io_enq_bits(rDataQo_Queue_1_io_enq_bits),
    .io_deq_ready(rDataQo_Queue_1_io_deq_ready),
    .io_deq_valid(rDataQo_Queue_1_io_deq_valid),
    .io_deq_bits(rDataQo_Queue_1_io_deq_bits)
  );
  Queue_0 wAddrQo_Queue_2 ( // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
    .clock(wAddrQo_Queue_2_clock),
    .reset(wAddrQo_Queue_2_reset),
    .io_enq_ready(wAddrQo_Queue_2_io_enq_ready),
    .io_enq_valid(wAddrQo_Queue_2_io_enq_valid),
    .io_enq_bits(wAddrQo_Queue_2_io_enq_bits),
    .io_deq_ready(wAddrQo_Queue_2_io_deq_ready),
    .io_deq_valid(wAddrQo_Queue_2_io_deq_valid),
    .io_deq_bits(wAddrQo_Queue_2_io_deq_bits)
  );
  Queue_3 wDataQo_Queue_3 ( // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
    .clock(wDataQo_Queue_3_clock),
    .reset(wDataQo_Queue_3_reset),
    .io_enq_ready(wDataQo_Queue_3_io_enq_ready),
    .io_enq_valid(wDataQo_Queue_3_io_enq_valid),
    .io_enq_bits_data(wDataQo_Queue_3_io_enq_bits_data),
    .io_deq_ready(wDataQo_Queue_3_io_deq_ready),
    .io_deq_valid(wDataQo_Queue_3_io_deq_valid),
    .io_deq_bits_data(wDataQo_Queue_3_io_deq_bits_data)
  );
  Queue_4 validAQ_Queue_4 ( // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
    .clock(validAQ_Queue_4_clock),
    .reset(validAQ_Queue_4_reset),
    .io_enq_ready(validAQ_Queue_4_io_enq_ready),
    .io_enq_valid(validAQ_Queue_4_io_enq_valid),
    .io_deq_ready(validAQ_Queue_4_io_deq_ready),
    .io_deq_valid(validAQ_Queue_4_io_deq_valid)
  );
  Queue_4 validDQ_Queue_5 ( // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 42:21]
    .clock(validDQ_Queue_5_clock),
    .reset(validDQ_Queue_5_reset),
    .io_enq_ready(validDQ_Queue_5_io_enq_ready),
    .io_enq_valid(validDQ_Queue_5_io_enq_valid),
    .io_deq_ready(validDQ_Queue_5_io_deq_ready),
    .io_deq_valid(validDQ_Queue_5_io_deq_valid)
  );
  assign S_AXI_LITE_awready = wAddrQi_ready & validAddress_ready; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 46:48]
  assign S_AXI_LITE_wready = wDataQi_ready & validData_ready; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 56:42]
  assign S_AXI_LITE_bvalid = rValidBuf; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 14:31 28:27]
  assign S_AXI_LITE_bresp = 2'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 14:31 29:26]
  assign S_AXI_LITE_arready = rAddressQo_Queue_0_io_enq_ready; // @[src/main/scala/xspn/fpga/streammapper/util/ChiselUtilities.scala 46:17 src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 59:26]
  assign S_AXI_LITE_rvalid = dValidBuf; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 78:27 94:23]
  assign S_AXI_LITE_rdata = dataBuf; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 78:27 95:22]
  assign S_AXI_LITE_rresp = 2'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 78:27 96:22]
  assign M_AXI_awvalid = 1'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 37:18]
  assign M_AXI_awid = 1'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 38:15]
  assign M_AXI_awaddr = 32'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 39:17]
  assign M_AXI_awlen = 8'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 40:16]
  assign M_AXI_awsize = 3'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 41:17]
  assign M_AXI_awburst = 2'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 42:18]
  assign M_AXI_awlock = 1'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 43:17]
  assign M_AXI_awcache = 4'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 44:18]
  assign M_AXI_awprot = 3'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 45:17]
  assign M_AXI_awqos = 4'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 46:16]
  assign M_AXI_awregion = 4'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 47:19]
  assign M_AXI_wvalid = 1'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 49:17]
  assign M_AXI_wdata = 512'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 50:16]
  assign M_AXI_wstrb = 64'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 51:16]
  assign M_AXI_wlast = 1'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 52:16]
  assign M_AXI_bready = 1'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 54:17]
  assign M_AXI_arvalid = 1'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 56:18]
  assign M_AXI_arid = 1'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 57:15]
  assign M_AXI_araddr = 32'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 58:17]
  assign M_AXI_arlen = 8'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 59:16]
  assign M_AXI_arsize = 3'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 60:17]
  assign M_AXI_arburst = 2'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 61:18]
  assign M_AXI_arlock = 1'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 62:17]
  assign M_AXI_arcache = 4'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 63:18]
  assign M_AXI_arprot = 3'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 64:17]
  assign M_AXI_arqos = 4'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 65:16]
  assign M_AXI_arregion = 4'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 66:19]
  assign M_AXI_rready = 1'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 68:17]
  assign interrupt = counter > 32'h0 ? 1'h0 : 1'h1; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 29:24 31:15 33:15]
  assign rAddressQo_Queue_0_clock = clock;
  assign rAddressQo_Queue_0_reset = reset;
  assign rAddressQo_Queue_0_io_enq_valid = rAddressQi_ready & S_AXI_LITE_arvalid; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 83:28]
  assign rAddressQo_Queue_0_io_enq_bits = rAddressQi_valid ? S_AXI_LITE_araddr : 32'h0; // @[src/main/scala/chisel3/util/Decoupled.scala 66:19 src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 82:22 83:53]
  assign rAddressQo_Queue_0_io_deq_ready = rAddressQo_Queue_0_io_deq_valid & rDataQi_ready; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 72:27]
  assign rDataQo_Queue_1_clock = clock;
  assign rDataQo_Queue_1_reset = reset;
  assign rDataQo_Queue_1_io_enq_valid = rAddressQo_Queue_0_io_deq_valid & rDataQi_ready; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 72:27]
  assign rDataQo_Queue_1_io_enq_bits = rDataQi_valid ? _GEN_43 : 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 72:45 75:11 71:23]
  assign rDataQo_Queue_1_io_deq_ready = ~dState & _GEN_3; // @[src/main/scala/chisel3/util/Decoupled.scala 90:20 src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 98:28]
  assign wAddrQo_Queue_2_clock = clock;
  assign wAddrQo_Queue_2_reset = reset;
  assign wAddrQo_Queue_2_io_enq_valid = wAddrQi_ready & S_AXI_LITE_awvalid; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 51:29]
  assign wAddrQo_Queue_2_io_enq_bits = wAddrQi_valid ? S_AXI_LITE_awaddr : 32'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 51:54 src/main/scala/chisel3/util/Decoupled.scala 66:19 src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 50:23]
  assign wAddrQo_Queue_2_io_deq_ready = wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:24]
  assign wDataQo_Queue_3_clock = clock;
  assign wDataQo_Queue_3_reset = reset;
  assign wDataQo_Queue_3_io_enq_valid = wDataQi_ready & S_AXI_LITE_wvalid; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 63:26]
  assign wDataQo_Queue_3_io_enq_bits_data = wDataQi_valid ? S_AXI_LITE_wdata : 32'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 63:48 src/main/scala/chisel3/util/Decoupled.scala 66:19 src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 60:25]
  assign wDataQo_Queue_3_io_deq_ready = wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:24]
  assign validAQ_Queue_4_clock = clock;
  assign validAQ_Queue_4_reset = reset;
  assign validAQ_Queue_4_io_enq_valid = wAddrQi_ready & S_AXI_LITE_awvalid; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 51:29]
  assign validAQ_Queue_4_io_deq_ready = ~rState & _T_5; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 31:28 src/main/scala/chisel3/util/Decoupled.scala 90:20]
  assign validDQ_Queue_5_clock = clock;
  assign validDQ_Queue_5_reset = reset;
  assign validDQ_Queue_5_io_enq_valid = wDataQi_ready & S_AXI_LITE_wvalid; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 63:26]
  assign validDQ_Queue_5_io_deq_ready = ~rState & _T_5; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 31:28 src/main/scala/chisel3/util/Decoupled.scala 90:20]
  always @(posedge clock) begin
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_0 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h0 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_0 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_0 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_1 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h1 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_1 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_1 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_2 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h2 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_2 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_2 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_3 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h3 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_3 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_3 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_4 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h4 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_4 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_4 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_5 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h5 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_5 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_5 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_6 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h6 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_6 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_6 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_7 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h7 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_7 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_7 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_8 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h8 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_8 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_8 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_9 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h9 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_9 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_9 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_10 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'ha == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_10 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_10 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_11 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'hb == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_11 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_11 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_12 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'hc == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_12 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_12 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_13 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'hd == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_13 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_13 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_14 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'he == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_14 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_14 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_15 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'hf == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_15 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_15 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_16 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h10 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_16 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_16 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_17 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h11 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_17 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_17 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_18 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h12 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_18 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_18 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_19 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h13 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_19 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_19 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_20 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h14 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_20 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_20 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_21 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h15 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_21 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_21 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_22 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h16 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_22 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_22 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_23 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h17 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_23 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_23 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_24 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h18 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_24 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_24 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_25 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h19 == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_25 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_25 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_26 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h1a == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_26 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_26 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
      registers_27 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 33:26]
    end else if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
      if (5'h1b == writeIdx[4:0]) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 95:27]
        if (wAddrQo_Queue_2_io_deq_valid & wDataQo_Queue_3_io_deq_valid) begin // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 92:42]
          registers_27 <= wDataQo_Queue_3_io_deq_bits_data; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 94:17]
        end else begin
          registers_27 <= 32'h0; // @[src/main/scala/xspn/fpga/streammapper/util/RegisterFileGen.scala 90:29]
        end
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 88:25]
      dState <= 1'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 88:25]
    end else if (~dState) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 98:28]
      dState <= _GEN_5;
    end else if (dState) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 104:35]
      if (S_AXI_LITE_rready) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 105:31]
        dState <= 1'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 108:16]
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 89:28]
      dValidBuf <= 1'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 89:28]
    end else if (~dState) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 98:28]
      dValidBuf <= _GEN_2;
    end else if (dState) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 104:35]
      if (S_AXI_LITE_rready) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 105:31]
        dValidBuf <= 1'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 106:19]
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 90:26]
      dataBuf <= 32'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 90:26]
    end else if (~dState) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 98:28]
      if (rDataQo_Queue_1_io_deq_valid) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 99:28]
        dataBuf <= rDataQo_Queue_1_io_deq_bits; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 101:17]
      end
    end else if (dState) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 104:35]
      if (S_AXI_LITE_rready) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 105:31]
        dataBuf <= 32'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 107:17]
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 23:25]
      rState <= 1'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 23:25]
    end else if (~rState) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 31:28]
      rState <= _GEN_50;
    end else if (rState) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 38:35]
      if (S_AXI_LITE_bready) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 39:35]
        rState <= 1'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 41:16]
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 24:28]
      rValidBuf <= 1'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 24:28]
    end else if (~rState) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 31:28]
      rValidBuf <= _GEN_48;
    end else if (rState) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 38:35]
      if (S_AXI_LITE_bready) begin // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 39:35]
        rValidBuf <= 1'h0; // @[src/main/scala/xspn/fpga/axi/AXI4LiteSlave.scala 40:19]
      end
    end
    if (reset) begin // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 23:24]
      counter <= 32'h0; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 23:24]
    end else if (counter > 32'h0) begin // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 29:24]
      counter <= _counter_T_1; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 30:13]
    end else if (registers_0[0]) begin // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 25:28]
      counter <= 32'h5; // @[src/main/scala/xspn/fpga/wrappers/DummyWrapper.scala 26:13]
    end
  end
// Register and memory initialization
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
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  registers_0 = _RAND_0[31:0];
  _RAND_1 = {1{`RANDOM}};
  registers_1 = _RAND_1[31:0];
  _RAND_2 = {1{`RANDOM}};
  registers_2 = _RAND_2[31:0];
  _RAND_3 = {1{`RANDOM}};
  registers_3 = _RAND_3[31:0];
  _RAND_4 = {1{`RANDOM}};
  registers_4 = _RAND_4[31:0];
  _RAND_5 = {1{`RANDOM}};
  registers_5 = _RAND_5[31:0];
  _RAND_6 = {1{`RANDOM}};
  registers_6 = _RAND_6[31:0];
  _RAND_7 = {1{`RANDOM}};
  registers_7 = _RAND_7[31:0];
  _RAND_8 = {1{`RANDOM}};
  registers_8 = _RAND_8[31:0];
  _RAND_9 = {1{`RANDOM}};
  registers_9 = _RAND_9[31:0];
  _RAND_10 = {1{`RANDOM}};
  registers_10 = _RAND_10[31:0];
  _RAND_11 = {1{`RANDOM}};
  registers_11 = _RAND_11[31:0];
  _RAND_12 = {1{`RANDOM}};
  registers_12 = _RAND_12[31:0];
  _RAND_13 = {1{`RANDOM}};
  registers_13 = _RAND_13[31:0];
  _RAND_14 = {1{`RANDOM}};
  registers_14 = _RAND_14[31:0];
  _RAND_15 = {1{`RANDOM}};
  registers_15 = _RAND_15[31:0];
  _RAND_16 = {1{`RANDOM}};
  registers_16 = _RAND_16[31:0];
  _RAND_17 = {1{`RANDOM}};
  registers_17 = _RAND_17[31:0];
  _RAND_18 = {1{`RANDOM}};
  registers_18 = _RAND_18[31:0];
  _RAND_19 = {1{`RANDOM}};
  registers_19 = _RAND_19[31:0];
  _RAND_20 = {1{`RANDOM}};
  registers_20 = _RAND_20[31:0];
  _RAND_21 = {1{`RANDOM}};
  registers_21 = _RAND_21[31:0];
  _RAND_22 = {1{`RANDOM}};
  registers_22 = _RAND_22[31:0];
  _RAND_23 = {1{`RANDOM}};
  registers_23 = _RAND_23[31:0];
  _RAND_24 = {1{`RANDOM}};
  registers_24 = _RAND_24[31:0];
  _RAND_25 = {1{`RANDOM}};
  registers_25 = _RAND_25[31:0];
  _RAND_26 = {1{`RANDOM}};
  registers_26 = _RAND_26[31:0];
  _RAND_27 = {1{`RANDOM}};
  registers_27 = _RAND_27[31:0];
  _RAND_28 = {1{`RANDOM}};
  dState = _RAND_28[0:0];
  _RAND_29 = {1{`RANDOM}};
  dValidBuf = _RAND_29[0:0];
  _RAND_30 = {1{`RANDOM}};
  dataBuf = _RAND_30[31:0];
  _RAND_31 = {1{`RANDOM}};
  rState = _RAND_31[0:0];
  _RAND_32 = {1{`RANDOM}};
  rValidBuf = _RAND_32[0:0];
  _RAND_33 = {1{`RANDOM}};
  counter = _RAND_33[31:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
