;buildInfoPackage: chisel3, version: 3.1.0, scalaVersion: 2.11.12, sbtVersion: 1.1.1, builtAtString: 2018-04-17 19:22:56.455, builtAtMillis: 1523992976455
circuit FPAdd : 
  module CompareComb : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip op1 : UInt<8>, flip op2 : UInt<8>, gte : UInt<1>}
    
    node _T_11 = geq(io.op1, io.op2) @[FPAdd.scala 72:20]
    io.gte <= _T_11 @[FPAdd.scala 72:10]
    
  module Swap : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip in_op1 : {m : UInt<23>, e : UInt<8>}, flip in_op2 : {m : UInt<23>, e : UInt<8>}, out_op1 : {m : UInt<23>, e : UInt<8>}, out_op2 : {m : UInt<23>, e : UInt<8>}}
    
    inst compare of CompareComb @[FPAdd.scala 83:23]
    compare.clock <= clock
    compare.reset <= reset
    compare.io.op1 <= io.in_op1.e @[FPAdd.scala 84:18]
    compare.io.op2 <= io.in_op2.e @[FPAdd.scala 85:18]
    node _T_5 = mux(compare.io.gte, io.in_op1, io.in_op2) @[FPAdd.scala 87:28]
    reg _T_7 : {m : UInt<23>, e : UInt<8>}, clock @[FPAdd.scala 87:24]
    _T_7.e <= _T_5.e @[FPAdd.scala 87:24]
    _T_7.m <= _T_5.m @[FPAdd.scala 87:24]
    io.out_op1.e <= _T_7.e @[FPAdd.scala 87:14]
    io.out_op1.m <= _T_7.m @[FPAdd.scala 87:14]
    node _T_8 = mux(compare.io.gte, io.in_op2, io.in_op1) @[FPAdd.scala 88:28]
    reg _T_10 : {m : UInt<23>, e : UInt<8>}, clock @[FPAdd.scala 88:24]
    _T_10.e <= _T_8.e @[FPAdd.scala 88:24]
    _T_10.m <= _T_8.m @[FPAdd.scala 88:24]
    io.out_op2.e <= _T_10.e @[FPAdd.scala 88:14]
    io.out_op2.m <= _T_10.m @[FPAdd.scala 88:14]
    
  module SubtractStage : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip minuend : UInt<8>, flip subtrahend : UInt<8>, difference : UInt<8>, minuend_is_zero : UInt<1>}
    
    node _T_13 = sub(io.minuend, io.subtrahend) @[FPAdd.scala 98:39]
    node _T_14 = asUInt(_T_13) @[FPAdd.scala 98:39]
    node _T_15 = tail(_T_14, 1) @[FPAdd.scala 98:39]
    reg difference : UInt, clock @[FPAdd.scala 98:27]
    difference <= _T_15 @[FPAdd.scala 98:27]
    io.difference <= difference @[FPAdd.scala 99:17]
    node _T_18 = eq(io.minuend, UInt<1>("h00")) @[FPAdd.scala 100:44]
    reg _T_20 : UInt<1>, clock @[FPAdd.scala 100:32]
    _T_20 <= _T_18 @[FPAdd.scala 100:32]
    io.minuend_is_zero <= _T_20 @[FPAdd.scala 100:22]
    
  module ShiftStage : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip value : UInt<23>, flip shamt : UInt<8>, out : UInt<24>}
    
    node _T_12 = cat(UInt<1>("h01"), io.value) @[Cat.scala 30:58]
    node _T_13 = dshr(_T_12, io.shamt) @[FPAdd.scala 109:44]
    reg shifted : UInt, clock @[FPAdd.scala 109:24]
    shifted <= _T_13 @[FPAdd.scala 109:24]
    io.out <= shifted @[FPAdd.scala 110:10]
    
  module PipelinedAdder : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : UInt<24>, flip b : UInt<24>, r : UInt<25>}
    
    node aPad = cat(UInt<8>("h00"), io.a) @[Cat.scala 30:58]
    node bPad = cat(UInt<8>("h00"), io.b) @[Cat.scala 30:58]
    node a_0 = bits(aPad, 31, 0) @[PipelinedAdder.scala 39:53]
    node b_0 = bits(bPad, 31, 0) @[PipelinedAdder.scala 40:53]
    node _T_13 = add(a_0, b_0) @[PipelinedAdder.scala 82:28]
    node _T_14 = add(_T_13, UInt<1>("h00")) @[PipelinedAdder.scala 82:33]
    node _T_15 = tail(_T_14, 1) @[PipelinedAdder.scala 82:33]
    reg _T_17 : UInt, clock @[PipelinedAdder.scala 82:25]
    _T_17 <= _T_15 @[PipelinedAdder.scala 82:25]
    node flushRes_0 = bits(_T_17, 31, 0) @[PipelinedAdder.scala 83:12]
    node carrySignals_1 = bits(_T_17, 32, 32) @[PipelinedAdder.scala 83:38]
    node _T_18 = cat(carrySignals_1, flushRes_0) @[Cat.scala 30:58]
    io.r <= _T_18 @[PipelinedAdder.scala 71:10]
    
  module MantissaShifterComb : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip sum : UInt<25>, out : UInt<23>, shifted : UInt<1>}
    
    node _T_11 = head(io.sum, 1) @[FPAdd.scala 119:36]
    node _T_12 = bits(_T_11, 0, 0) @[FPAdd.scala 119:46]
    node _T_13 = tail(io.sum, 1) @[FPAdd.scala 120:16]
    node _T_14 = head(_T_13, 23) @[FPAdd.scala 120:24]
    node _T_15 = tail(io.sum, 2) @[FPAdd.scala 121:16]
    node _T_16 = head(_T_15, 23) @[FPAdd.scala 121:24]
    node shifted_sum = mux(_T_12, _T_14, _T_16) @[FPAdd.scala 119:24]
    node shifted = head(io.sum, 1) @[FPAdd.scala 122:28]
    io.out <= shifted_sum @[FPAdd.scala 123:10]
    io.shifted <= shifted @[FPAdd.scala 124:14]
    
  module ExponentAdder : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip e_in : UInt<8>, flip add : UInt<1>, flip inputs_are_zero : UInt<1>, e_out : UInt<8>}
    
    node _T_14 = eq(io.inputs_are_zero, UInt<1>("h00")) @[FPAdd.scala 134:35]
    node _T_15 = and(io.add, _T_14) @[FPAdd.scala 134:32]
    node _T_17 = add(io.e_in, UInt<1>("h01")) @[FPAdd.scala 134:64]
    node _T_18 = tail(_T_17, 1) @[FPAdd.scala 134:64]
    node _T_19 = mux(_T_15, _T_18, io.e_in) @[FPAdd.scala 134:24]
    reg out : UInt, clock @[FPAdd.scala 134:20]
    out <= _T_19 @[FPAdd.scala 134:20]
    io.e_out <= out @[FPAdd.scala 135:12]
    
  module FPAdd : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : UInt<31>, flip b : UInt<31>, r : UInt<31>}
    
    reg op1_packed : UInt, clock @[FPAdd.scala 10:27]
    op1_packed <= io.a @[FPAdd.scala 10:27]
    reg op2_packed : UInt, clock @[FPAdd.scala 11:27]
    op2_packed <= io.b @[FPAdd.scala 11:27]
    wire op1 : {m : UInt<23>, e : UInt<8>} @[UFloat.scala 18:22]
    node _T_14 = head(op1_packed, 8) @[UFloat.scala 19:26]
    op1.e <= _T_14 @[UFloat.scala 19:14]
    node _T_15 = tail(op1_packed, 8) @[UFloat.scala 20:26]
    op1.m <= _T_15 @[UFloat.scala 20:14]
    wire op2 : {m : UInt<23>, e : UInt<8>} @[UFloat.scala 18:22]
    node _T_17 = head(op2_packed, 8) @[UFloat.scala 19:26]
    op2.e <= _T_17 @[UFloat.scala 19:14]
    node _T_18 = tail(op2_packed, 8) @[UFloat.scala 20:26]
    op2.m <= _T_18 @[UFloat.scala 20:14]
    inst swap of Swap @[FPAdd.scala 17:20]
    swap.clock <= clock
    swap.reset <= reset
    swap.io.in_op1.e <= op1.e @[FPAdd.scala 18:18]
    swap.io.in_op1.m <= op1.m @[FPAdd.scala 18:18]
    swap.io.in_op2.e <= op2.e @[FPAdd.scala 19:18]
    swap.io.in_op2.m <= op2.m @[FPAdd.scala 19:18]
    inst subtractStage of SubtractStage @[FPAdd.scala 22:29]
    subtractStage.clock <= clock
    subtractStage.reset <= reset
    subtractStage.io.minuend <= swap.io.out_op1.e @[FPAdd.scala 23:28]
    subtractStage.io.subtrahend <= swap.io.out_op2.e @[FPAdd.scala 24:31]
    reg m1_2 : UInt, clock @[FPAdd.scala 26:21]
    m1_2 <= swap.io.out_op1.m @[FPAdd.scala 26:21]
    reg m2_2 : UInt, clock @[FPAdd.scala 27:21]
    m2_2 <= swap.io.out_op2.m @[FPAdd.scala 27:21]
    reg e1_2 : UInt, clock @[FPAdd.scala 28:21]
    e1_2 <= swap.io.out_op1.e @[FPAdd.scala 28:21]
    reg e2_2 : UInt, clock @[FPAdd.scala 29:21]
    e2_2 <= swap.io.out_op2.e @[FPAdd.scala 29:21]
    inst shiftStage of ShiftStage @[FPAdd.scala 32:26]
    shiftStage.clock <= clock
    shiftStage.reset <= reset
    shiftStage.io.value <= m2_2 @[FPAdd.scala 33:23]
    shiftStage.io.shamt <= subtractStage.io.difference @[FPAdd.scala 34:23]
    reg m1_3 : UInt, clock @[FPAdd.scala 35:21]
    m1_3 <= m1_2 @[FPAdd.scala 35:21]
    reg e1_3 : UInt, clock @[FPAdd.scala 37:21]
    e1_3 <= e1_2 @[FPAdd.scala 37:21]
    reg minuend_is_zero_3 : UInt<1>, clock @[FPAdd.scala 38:34]
    minuend_is_zero_3 <= subtractStage.io.minuend_is_zero @[FPAdd.scala 38:34]
    node a = cat(UInt<1>("h01"), m1_3) @[Cat.scala 30:58]
    inst PipelinedAdder of PipelinedAdder @[PipelinedAdder.scala 98:21]
    PipelinedAdder.clock <= clock
    PipelinedAdder.reset <= reset
    PipelinedAdder.io.a <= a @[PipelinedAdder.scala 99:14]
    PipelinedAdder.io.b <= shiftStage.io.out @[PipelinedAdder.scala 100:14]
    reg minuend_is_zero_4 : UInt<1>, clock @[Reg.scala 11:16]
    when UInt<1>("h01") : @[Reg.scala 12:19]
      minuend_is_zero_4 <= minuend_is_zero_3 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg e1_4 : UInt, clock @[Reg.scala 11:16]
    when UInt<1>("h01") : @[Reg.scala 12:19]
      e1_4 <= e1_3 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    inst mantissaShifterComb of MantissaShifterComb @[FPAdd.scala 48:35]
    mantissaShifterComb.clock <= clock
    mantissaShifterComb.reset <= reset
    mantissaShifterComb.io.sum <= PipelinedAdder.io.r @[FPAdd.scala 49:30]
    inst exponentAdder of ExponentAdder @[FPAdd.scala 52:29]
    exponentAdder.clock <= clock
    exponentAdder.reset <= reset
    exponentAdder.io.e_in <= e1_4 @[FPAdd.scala 53:25]
    exponentAdder.io.add <= mantissaShifterComb.io.shifted @[FPAdd.scala 54:24]
    exponentAdder.io.inputs_are_zero <= minuend_is_zero_4 @[FPAdd.scala 55:36]
    reg m_6 : UInt, clock @[FPAdd.scala 56:20]
    m_6 <= mantissaShifterComb.io.out @[FPAdd.scala 56:20]
    wire out : {m : UInt<23>, e : UInt<8>} @[FPAdd.scala 59:17]
    out.m <= m_6 @[FPAdd.scala 60:9]
    out.e <= exponentAdder.io.e_out @[FPAdd.scala 61:9]
    node output_together = cat(out.e, out.m) @[Cat.scala 30:58]
    io.r <= output_together @[FPAdd.scala 63:8]
    
