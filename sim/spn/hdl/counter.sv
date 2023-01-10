module counter(input clk, output [7:0] value);

  logic [7:0] state;
  logic [7:0] next;

  assign next = state + 1;
  assign value = state;

  always_ff @(posedge clk) begin
    state <= next;
  end

  initial begin
    state = 0;
  end

endmodule
