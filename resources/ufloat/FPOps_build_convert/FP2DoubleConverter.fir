;buildInfoPackage: chisel3, version: 3.1.0, scalaVersion: 2.11.12, sbtVersion: 1.1.1, builtAtString: 2018-04-17 19:22:56.455, builtAtMillis: 1523992976455
circuit FP2DoubleConverter : 
  module FP2DoubleConverter : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip op : UInt<31>, r : UInt<64>}
    
    wire inputs : {m : UInt<23>, e : UInt<8>} @[FP2DoubleConverter.scala 9:20]
    node _T_10 = head(io.op, 8) @[FP2DoubleConverter.scala 10:33]
    reg _T_12 : UInt, clock @[FP2DoubleConverter.scala 10:22]
    _T_12 <= _T_10 @[FP2DoubleConverter.scala 10:22]
    inputs.e <= _T_12 @[FP2DoubleConverter.scala 10:12]
    node _T_13 = tail(io.op, 8) @[FP2DoubleConverter.scala 11:33]
    reg _T_15 : UInt, clock @[FP2DoubleConverter.scala 11:22]
    _T_15 <= _T_13 @[FP2DoubleConverter.scala 11:22]
    inputs.m <= _T_15 @[FP2DoubleConverter.scala 11:12]
    node new_mantissa = shl(inputs.m, 29) @[FP2DoubleConverter.scala 19:31]
    node new_exponent_nonzero = add(inputs.e, UInt<10>("h0380")) @[FP2DoubleConverter.scala 20:39]
    node _T_18 = eq(inputs.e, UInt<1>("h00")) @[FP2DoubleConverter.scala 21:30]
    node _T_20 = eq(inputs.m, UInt<1>("h00")) @[FP2DoubleConverter.scala 21:50]
    node inputs_zero = and(_T_18, _T_20) @[FP2DoubleConverter.scala 21:38]
    node new_exponent = mux(inputs_zero, UInt<1>("h00"), new_exponent_nonzero) @[FP2DoubleConverter.scala 22:25]
    wire output : {m : UInt<52>, e : UInt<11>} @[FP2DoubleConverter.scala 24:20]
    output.e <= new_exponent @[FP2DoubleConverter.scala 25:12]
    output.m <= new_mantissa @[FP2DoubleConverter.scala 26:12]
    reg res : {m : UInt<52>, e : UInt<11>}, clock @[FP2DoubleConverter.scala 27:20]
    res.e <= output.e @[FP2DoubleConverter.scala 27:20]
    res.m <= output.m @[FP2DoubleConverter.scala 27:20]
    node _T_24 = cat(res.e, res.m) @[Cat.scala 30:58]
    io.r <= _T_24 @[FP2DoubleConverter.scala 29:8]
    
