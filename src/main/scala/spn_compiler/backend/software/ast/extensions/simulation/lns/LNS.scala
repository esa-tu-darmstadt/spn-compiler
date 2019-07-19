package spn_compiler.backend.software.ast.extensions.simulation.lns

import java.util.concurrent.ThreadLocalRandom

import scala.collection.mutable.ListBuffer

class LNS(val intSize: Int, val fracSize: Int) {
  var zero: Boolean = false
  var sign: Boolean = false
  var exp: ListBuffer[Boolean] = ListBuffer.fill(intSize + fracSize)(false)

  def toLNSDouble: Double = {
    var double = 0.0
    var current = Math.pow(2, intSize - 1)
    for (e <- exp) {
      if (e) double = double + current
      current = current / 2
    }
    double
  }

  def toDouble: Double = {
    val double = toLNSDouble
    if (zero) 0.0 else if (sign) Math.pow(2, -double) else Math.pow(2, double)
  }

  def toBigInt: BigInt = {
    var big = BigInt(new String(Array.fill[Char](fracSize + intSize + 2)("0".charAt(0))))
    if (zero) {
      big.setBit(intSize + fracSize + 2 - 1)
    } else {
      if (sign) big = big.setBit(intSize + fracSize + 2 - 2)
      var c = intSize + fracSize + 2 - 3
      for (i <- exp) {
        if (i) big = big.setBit(c)
        c = c - 1
      }
      big
    }
  }

  def exp2BigInt: BigInt = {
    var big = BigInt(new String(Array.fill[Char](fracSize + intSize)("0".charAt(0))))
    var c = intSize + fracSize + 2 - 3
    for (i <- exp) {
      if (i) big = big.setBit(c)
      c = c - 1
    }
    big
  }

  override def toString: String = {
    val sb = new StringBuilder
    sb.append(bool2char(zero)).append(bool2char(sign))
    exp.slice(0, intSize).map(x => sb.append(bool2char(x)))
    exp.slice(intSize, intSize + fracSize).map(x => sb.append(bool2char(x)))
    sb.toString()
  }

  def prettyString: String = {
    val sb = new StringBuilder
    sb.append(bool2char(zero)).append(bool2char(sign)).append(" ")
    exp.slice(0, intSize).map(x => sb.append(bool2char(x)))
    sb.append(" ")
    exp.slice(intSize, intSize + fracSize).map(x => sb.append(bool2char(x)))
    sb.toString()
  }

  def bool2char(boolean: Boolean): String = if (boolean) "1" else "0"

  override def equals(o: Any): Boolean = o match {
    case other: LNS => {
      if (intSize != other.intSize) false
      if (fracSize != other.fracSize) false
      if (!(sign ^ other.sign)) false
      if (!(zero ^ other.zero)) false
      exp.equals(other.exp)
    }
    case _ => false
  }

}

object LNS {

  /**
    * Returns a LNS which has all bits unset, used for testing purposes.
    * @param intSize integer bits
    * @param fracSize fractional bits
    * @return LNS
    */
  def test(intSize: Int, fracSize: Int): LNS = {
    new LNS(intSize, fracSize)
  }

  def random(intSize: Int, fracSize: Int): LNS = {
    val rand = ThreadLocalRandom.current().nextDouble(0.0, Double.MaxValue)
    LNS(rand, intSize, fracSize)
  }

  def apply(string: String, intSize: Int, fracSize: Int): LNS = {
    val lns = new LNS(intSize, fracSize)
    var str = string
    require(str.length <= fracSize + intSize + 2)
    while (str.length < fracSize + intSize + 2) {
      str = "0" + str
    }
    val lst = str.toCharArray().map(x => x == "1".charAt(0))
    if (lst(0)) lns.zero = true
    if (lst(1)) lns.sign = true
    lns.exp = lst.slice(2, lst.length).to
    lns
  }

  def apply(bigInt: BigInt, intSize: Int, fracSize: Int): LNS = {
    apply(bigInt.toString(2), intSize, fracSize)
  }

  def apply(double: Double, intSize: Int, fracSize: Int, bool: Boolean): LNS = {
    val lns = new LNS(intSize, fracSize)
    if (double == 0.0) {
      lns.zero = true
    } else {
      var current = Math.pow(2, intSize - 1)
      var log2 = double
      lns.sign = log2 < 0.0
      log2 = Math.abs(log2)
      var exp = ListBuffer[Boolean]()
      for (i <- 0 until intSize + fracSize) {
        exp += current <= log2
        if (current <= log2) log2 = log2 - current
        current = current / 2.0
      }
      lns.exp = exp
    }
    lns
  }

  def apply(double: Double, intSize: Int, fracSize: Int): LNS = {
    val lns = new LNS(intSize, fracSize)
    if (double == 0.0) {
      lns.zero = true
    } else {
      var current = Math.pow(2, intSize - 1)
      var log2 = Math.log(double) / Math.log(2.0)
      lns.sign = log2 < 0.0
      log2 = Math.abs(log2)
      var exp = ListBuffer[Boolean]()
      for (i <- 0 until intSize + fracSize) {
        exp += current <= log2
        if (current <= log2) log2 = log2 - current
        current = current / 2.0
      }
      lns.exp = exp
    }
    lns
  }

  def apply(float: Float, intSize: Int, fracSize: Int): LNS = apply(float.toDouble, intSize, fracSize)

  def double2BigInt(double: Double, int: Int, frac: Int): BigInt = apply(double,int,frac, true).toBigInt

  def exp2BigInt(double: Double, int: Int, frac: Int): BigInt = apply(double, int, frac, true).exp2BigInt

  def roundedExp(target: Double, int: Int, frac: Int): BigInt = {
    val lns = LNS(target, int, frac, true)
    val a = lns.exp2BigInt
    val b = a + 1
    val c = a - 1

    val errA = Math.abs(LNS(a, int, frac).toLNSDouble - target)
    val errB = Math.abs(LNS(b, int, frac).toLNSDouble - target)
    val errC = Math.abs(LNS(c, int, frac).toLNSDouble - target)

    val min = Math.min(errC, Math.min(errA, errB))
    if(errA == min) a
    else if (errB == min) b
    else c
  }

  def LNSDoubleFromBigInt(in: BigInt): Double = {
    val s = in.toString(2)
    if(s.length > 63) {
      -1 * java.lang.Double.longBitsToDouble(java.lang.Long.parseLong(s.substring(1), 2))
    } else {
      java.lang.Double.longBitsToDouble(java.lang.Long.parseLong(s, 2))
    }
  }
}