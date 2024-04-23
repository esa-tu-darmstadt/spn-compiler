;buildInfoPackage: chisel3, version: 3.1.0, scalaVersion: 2.11.12, sbtVersion: 1.1.1, builtAtString: 2018-04-17 19:22:56.455, builtAtMillis: 1523992976455
circuit FPMult : 
  module DSPMult24x17 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : UInt<24>, flip b : UInt<17>, r : UInt<41>}
    
    node _T_11 = mul(io.a, io.b) @[DSPMult.scala 70:27]
    reg rOut : UInt, clock @[DSPMult.scala 70:21]
    rOut <= _T_11 @[DSPMult.scala 70:21]
    io.r <= rOut @[DSPMult.scala 71:8]
    
  module DSPMult24x17_1 : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : UInt<24>, flip b : UInt<17>, r : UInt<41>}
    
    node _T_11 = mul(io.a, io.b) @[DSPMult.scala 70:27]
    reg rOut : UInt, clock @[DSPMult.scala 70:21]
    rOut <= _T_11 @[DSPMult.scala 70:21]
    io.r <= rOut @[DSPMult.scala 71:8]
    
  module PipelinedAdder : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : UInt<48>, flip b : UInt<48>, r : UInt<49>}
    
    node aPad = cat(UInt<16>("h00"), io.a) @[Cat.scala 30:58]
    node bPad = cat(UInt<16>("h00"), io.b) @[Cat.scala 30:58]
    node a_0 = bits(aPad, 31, 0) @[PipelinedAdder.scala 39:53]
    node a_1 = bits(aPad, 63, 32) @[PipelinedAdder.scala 39:53]
    node b_0 = bits(bPad, 31, 0) @[PipelinedAdder.scala 40:53]
    node b_1 = bits(bPad, 63, 32) @[PipelinedAdder.scala 40:53]
    reg aDelayed_1 : UInt<32>, clock @[Reg.scala 11:16]
    when UInt<1>("h01") : @[Reg.scala 12:19]
      aDelayed_1 <= a_1 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg bDelayed_1 : UInt<32>, clock @[Reg.scala 11:16]
    when UInt<1>("h01") : @[Reg.scala 12:19]
      bDelayed_1 <= b_1 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    node _T_17 = add(a_0, b_0) @[PipelinedAdder.scala 82:28]
    node _T_18 = add(_T_17, UInt<1>("h00")) @[PipelinedAdder.scala 82:33]
    node _T_19 = tail(_T_18, 1) @[PipelinedAdder.scala 82:33]
    reg _T_21 : UInt, clock @[PipelinedAdder.scala 82:25]
    _T_21 <= _T_19 @[PipelinedAdder.scala 82:25]
    node resultSignals_0 = bits(_T_21, 31, 0) @[PipelinedAdder.scala 83:12]
    node carrySignals_1 = bits(_T_21, 32, 32) @[PipelinedAdder.scala 83:38]
    node _T_22 = add(aDelayed_1, bDelayed_1) @[PipelinedAdder.scala 82:28]
    node _T_23 = add(_T_22, carrySignals_1) @[PipelinedAdder.scala 82:33]
    node _T_24 = tail(_T_23, 1) @[PipelinedAdder.scala 82:33]
    reg _T_26 : UInt, clock @[PipelinedAdder.scala 82:25]
    _T_26 <= _T_24 @[PipelinedAdder.scala 82:25]
    node flushRes_1 = bits(_T_26, 31, 0) @[PipelinedAdder.scala 83:12]
    node carrySignals_2 = bits(_T_26, 32, 32) @[PipelinedAdder.scala 83:38]
    reg flushRes_0 : UInt<32>, clock @[Reg.scala 11:16]
    when UInt<1>("h01") : @[Reg.scala 12:19]
      flushRes_0 <= resultSignals_0 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    node _T_29 = cat(flushRes_1, flushRes_0) @[Cat.scala 30:58]
    node _T_30 = cat(carrySignals_2, _T_29) @[Cat.scala 30:58]
    io.r <= _T_30 @[PipelinedAdder.scala 73:10]
    
  module DSPMult : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : UInt<24>, flip b : UInt<24>, r : UInt<48>}
    
    node _T_11 = bits(io.a, 23, 0) @[DSPMult.scala 113:16]
    node _T_12 = bits(io.b, 23, 17) @[DSPMult.scala 114:16]
    inst DSPMult24x17 of DSPMult24x17 @[DSPMult.scala 178:22]
    DSPMult24x17.clock <= clock
    DSPMult24x17.reset <= reset
    DSPMult24x17.io.a <= _T_11 @[DSPMult.scala 179:15]
    DSPMult24x17.io.b <= _T_12 @[DSPMult.scala 180:15]
    node _T_14 = cat(DSPMult24x17.io.r, UInt<17>("h00")) @[Cat.scala 30:58]
    node _T_15 = bits(io.a, 23, 0) @[DSPMult.scala 113:16]
    node _T_16 = bits(io.b, 16, 0) @[DSPMult.scala 114:16]
    inst DSPMult24x17_1 of DSPMult24x17_1 @[DSPMult.scala 178:22]
    DSPMult24x17_1.clock <= clock
    DSPMult24x17_1.reset <= reset
    DSPMult24x17_1.io.a <= _T_15 @[DSPMult.scala 179:15]
    DSPMult24x17_1.io.b <= _T_16 @[DSPMult.scala 180:15]
    inst PipelinedAdder of PipelinedAdder @[PipelinedAdder.scala 98:21]
    PipelinedAdder.clock <= clock
    PipelinedAdder.reset <= reset
    PipelinedAdder.io.a <= _T_14 @[PipelinedAdder.scala 99:14]
    PipelinedAdder.io.b <= DSPMult24x17_1.io.r @[PipelinedAdder.scala 100:14]
    io.r <= PipelinedAdder.io.r @[DSPMult.scala 141:8]
    
  module ZeroCheckComb : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip op1 : {m : UInt<23>, e : UInt<8>}, flip op2 : {m : UInt<23>, e : UInt<8>}, out : UInt<1>}
    
    node _T_12 = eq(io.op1.m, UInt<1>("h00")) @[FPMult.scala 53:25]
    node _T_14 = eq(io.op1.e, UInt<1>("h00")) @[FPMult.scala 53:45]
    node _T_15 = and(_T_12, _T_14) @[FPMult.scala 53:33]
    node _T_17 = eq(io.op2.m, UInt<1>("h00")) @[FPMult.scala 53:65]
    node _T_19 = eq(io.op2.e, UInt<1>("h00")) @[FPMult.scala 53:85]
    node _T_20 = and(_T_17, _T_19) @[FPMult.scala 53:73]
    node output = or(_T_15, _T_20) @[FPMult.scala 53:53]
    io.out <= output @[FPMult.scala 54:10]
    
  module AddExponentsComb : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip e1 : UInt<8>, flip e2 : UInt<8>, eout : UInt<9>}
    
    node _T_11 = add(io.e1, io.e2) @[FPMult.scala 63:27]
    reg out : UInt, clock @[FPMult.scala 63:20]
    out <= _T_11 @[FPMult.scala 63:20]
    io.eout <= out @[FPMult.scala 64:11]
    
  module AddOneToExpAndSubtractOffsetComb : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip e1 : UInt<9>, flip add : UInt<1>, e_out : UInt<8>, underflow_out : UInt<1>}
    
    node _T_14 = cat(UInt<1>("h00"), io.e1) @[Cat.scala 30:58]
    node e1_s = asSInt(_T_14) @[FPMult.scala 76:36]
    node _T_15 = bits(io.add, 0, 0) @[FPMult.scala 77:42]
    node _T_17 = sub(e1_s, asSInt(UInt<8>("h07e"))) @[FPMult.scala 77:51]
    node _T_18 = tail(_T_17, 1) @[FPMult.scala 77:51]
    node _T_19 = asSInt(_T_18) @[FPMult.scala 77:51]
    node _T_21 = sub(e1_s, asSInt(UInt<8>("h07f"))) @[FPMult.scala 77:72]
    node _T_22 = tail(_T_21, 1) @[FPMult.scala 77:72]
    node _T_23 = asSInt(_T_22) @[FPMult.scala 77:72]
    node selected = mux(_T_15, _T_19, _T_23) @[FPMult.scala 77:28]
    node _T_24 = head(selected, 1) @[FPMult.scala 78:36]
    io.underflow_out <= _T_24 @[FPMult.scala 78:20]
    node _T_25 = tail(selected, 2) @[FPMult.scala 79:28]
    io.e_out <= _T_25 @[FPMult.scala 79:12]
    
  module MantissaSelectorComb : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip m_in : UInt<48>, m_out : UInt<23>}
    
    node _T_9 = head(io.m_in, 1) @[FPMult.scala 87:34]
    node _T_10 = bits(_T_9, 0, 0) @[FPMult.scala 87:44]
    node _T_11 = tail(io.m_in, 1) @[FPMult.scala 88:34]
    node _T_12 = head(_T_11, 23) @[FPMult.scala 88:42]
    node _T_13 = tail(io.m_in, 2) @[FPMult.scala 89:34]
    node _T_14 = head(_T_13, 23) @[FPMult.scala 89:42]
    node out = mux(_T_10, _T_12, _T_14) @[FPMult.scala 87:21]
    io.m_out <= out @[FPMult.scala 91:12]
    
  module ZeroHandler : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip m_in : UInt<23>, flip e_in : UInt<8>, flip underflow_in : UInt<1>, flip zero : UInt<1>, out : {m : UInt<23>, e : UInt<8>}}
    
    node _T_15 = or(io.zero, io.underflow_in) @[FPMult.scala 102:35]
    node _T_17 = mux(_T_15, UInt<1>("h00"), io.m_in) @[FPMult.scala 102:26]
    reg m_out : UInt, clock @[FPMult.scala 102:22]
    m_out <= _T_17 @[FPMult.scala 102:22]
    node _T_19 = or(io.zero, io.underflow_in) @[FPMult.scala 103:35]
    node _T_21 = mux(_T_19, UInt<1>("h00"), io.e_in) @[FPMult.scala 103:26]
    reg e_out : UInt, clock @[FPMult.scala 103:22]
    e_out <= _T_21 @[FPMult.scala 103:22]
    io.out.m <= m_out @[FPMult.scala 104:12]
    io.out.e <= e_out @[FPMult.scala 105:12]
    
  module FPMult : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip a : UInt<31>, flip b : UInt<31>, r : UInt<31>}
    
    reg op1_packed : UInt, clock @[FPMult.scala 11:27]
    op1_packed <= io.a @[FPMult.scala 11:27]
    reg op2_packed : UInt, clock @[FPMult.scala 12:27]
    op2_packed <= io.b @[FPMult.scala 12:27]
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
    inst multiplyMantissas of DSPMult @[FPMult.scala 18:33]
    multiplyMantissas.clock <= clock
    multiplyMantissas.reset <= reset
    node _T_20 = cat(UInt<1>("h01"), op1.m) @[Cat.scala 30:58]
    multiplyMantissas.io.a <= _T_20 @[FPMult.scala 19:26]
    node _T_22 = cat(UInt<1>("h01"), op2.m) @[Cat.scala 30:58]
    multiplyMantissas.io.b <= _T_22 @[FPMult.scala 20:26]
    inst zeroCheck of ZeroCheckComb @[FPMult.scala 21:25]
    zeroCheck.clock <= clock
    zeroCheck.reset <= reset
    zeroCheck.io.op1.e <= op1.e @[FPMult.scala 22:20]
    zeroCheck.io.op1.m <= op1.m @[FPMult.scala 22:20]
    zeroCheck.io.op2.e <= op2.e @[FPMult.scala 23:20]
    zeroCheck.io.op2.m <= op2.m @[FPMult.scala 23:20]
    inst addExponentsWithOffset of AddExponentsComb @[FPMult.scala 25:38]
    addExponentsWithOffset.clock <= clock
    addExponentsWithOffset.reset <= reset
    addExponentsWithOffset.io.e1 <= op1.e @[FPMult.scala 26:32]
    addExponentsWithOffset.io.e2 <= op2.e @[FPMult.scala 27:32]
    inst addOne of AddOneToExpAndSubtractOffsetComb @[FPMult.scala 31:22]
    addOne.clock <= clock
    addOne.reset <= reset
    reg _T_25 : UInt<9>, clock @[Reg.scala 11:16]
    when UInt<1>("h01") : @[Reg.scala 12:19]
      _T_25 <= addExponentsWithOffset.io.eout @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_27 : UInt<9>, clock @[Reg.scala 11:16]
    when UInt<1>("h01") : @[Reg.scala 12:19]
      _T_27 <= _T_25 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    addOne.io.e1 <= _T_27 @[FPMult.scala 32:16]
    node _T_28 = head(multiplyMantissas.io.r, 1) @[FPMult.scala 33:47]
    addOne.io.add <= _T_28 @[FPMult.scala 33:17]
    inst mantissaSelector of MantissaSelectorComb @[FPMult.scala 34:32]
    mantissaSelector.clock <= clock
    mantissaSelector.reset <= reset
    mantissaSelector.io.m_in <= multiplyMantissas.io.r @[FPMult.scala 35:28]
    inst zeroHandler of ZeroHandler @[FPMult.scala 37:27]
    zeroHandler.clock <= clock
    zeroHandler.reset <= reset
    zeroHandler.io.m_in <= mantissaSelector.io.m_out @[FPMult.scala 38:23]
    zeroHandler.io.e_in <= addOne.io.e_out @[FPMult.scala 39:23]
    zeroHandler.io.underflow_in <= addOne.io.underflow_out @[FPMult.scala 40:31]
    reg _T_31 : UInt<1>, clock @[Reg.scala 11:16]
    when UInt<1>("h01") : @[Reg.scala 12:19]
      _T_31 <= zeroCheck.io.out @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    reg _T_33 : UInt<1>, clock @[Reg.scala 11:16]
    when UInt<1>("h01") : @[Reg.scala 12:19]
      _T_33 <= _T_31 @[Reg.scala 12:23]
      skip @[Reg.scala 12:19]
    zeroHandler.io.zero <= _T_33 @[FPMult.scala 41:23]
    node _T_34 = cat(zeroHandler.io.out.e, zeroHandler.io.out.m) @[Cat.scala 30:58]
    io.r <= _T_34 @[FPMult.scala 44:8]
    
