package spn_compiler.driver.compile.cpu.headers

import java.io.{BufferedWriter, File, FileWriter}

object CPPLNSHeader {

  def writeHeader(file: File) : Unit = {
    val writer = new BufferedWriter(new FileWriter(file))
    writer.write(headerCode)
    writer.close()
  }

  private val headerCode =
    """#ifndef LNSOFT_LNS_H
      |#define LNSOFT_LNS_H
      |
      |#include <tuple>
      |#include <vector>
      |#include <bitset>
      |#include <assert.h>
      |#include <cmath>
      |#include <iostream>
      |
      |#ifndef LNS_INTEGER_BITS
      |  #define LNS_INTEGER_BITS 8
      |#endif
      |
      |#ifndef LNS_FRACTION_BITS
      |  #define LNS_FRACTION_BITS 32
      |#endif
      |
      |inline __int128 bitmask(uint32_t bits) {
      |    __int128 result = 0;
      |    for (uint32_t i = 0; i < bits; i++) {
      |        result = (result << 1) + 1;
      |    }
      |    return result;
      |}
      |
      |inline bool checkOverflow(__int128 result, uint32_t bits) {
      |    auto mask = ~bitmask(bits);
      |
      |    result = (result & mask) >> bits;
      |    if (result != 0) return true;
      |    else return false;
      |}
      |
      |class Interpolator{
      |private:
      |    std::vector<__int128> memA, memB, memC;
      |    uint32_t upperAddressBit, lowerAddressBit, numAddressBits;
      |    uint32_t integer,fraction;
      |    __int128 lastInterpolated = 0;
      |
      |public:
      |
      |    Interpolator(uint32_t integer, uint32_t fraction, std::vector<__int128> memA, std::vector<__int128> memB,
      |                        std::vector<__int128> memC, uint32_t upperAddressBit, uint32_t lowerAddressBit,
      |                        uint32_t numAddressBits, __int128 last) : memA(std::move(memA)), memB(std::move(memB)), memC(std::move(memC)), upperAddressBit(upperAddressBit),
      |                                                   lowerAddressBit(lowerAddressBit), numAddressBits(numAddressBits), integer(integer), fraction(fraction),
      |                                                   lastInterpolated(last) {
      |        assert(integer + fraction >= upperAddressBit - 1);
      |    };
      |
      |    __int128 ip(__int128 x) {
      |        uint32_t address = getAddress(x);
      |        if(address == -1 ) return 0;
      |        __int128 a = memA[address];
      |        __int128 b = memB[address];
      |        __int128 c = memC[address];
      |
      |        __int128 ax2 = ((x * x * a) >> (fraction * 2)) & bitmask((integer + fraction) * 2);
      |        __int128 bx = ((x * b) >> (fraction)) & bitmask(integer + fraction);
      |        __int128 result = (ax2 + c - bx) & bitmask(integer + fraction);
      |
      |        if(checkOverflow(result, integer + fraction)) {
      |            return 0;
      |        } else return result;
      |    }
      |
      |    uint32_t getAddress(__int128 x) {
      |        if(x > lastInterpolated) {
      |            return -1;
      |        }
      |
      |        __int128 mask = bitmask(numAddressBits);
      |        __int128 shifted = x >> lowerAddressBit;
      |        return shifted & mask;
      |    }
      |
      |};
      |
      |static Interpolator* interpolator;
      |
      |struct QuadSpline {
      |    double_t x,y,a,b,c,h = 0.0;
      |
      |    QuadSpline(double_t x, double_t y, double_t a, double_t b, double_t c, double_t h = 0.0)
      |            : x(x), y(y), a(a), b(b), c(c), h(h) {};
      |
      |    bool valid(double_t i) {
      |        return i <= x && x <= x+h && h != 0.0;
      |    }
      |
      |    double_t interpolate(double_t i) {
      |        return i*i*a + i*b + c;
      |    }
      |};
      |
      |inline __int128 value2fixedInt(double_t value, uint32_t integerBits, uint32_t fractionBits) {
      |    value = fabs(value);
      |    __int128 result = (__int128)floor(value) << fractionBits;
      |    double_t fractional = value - floor(value);
      |
      |    for(int currentExp = 1; currentExp <= fractionBits; currentExp++) {
      |        double_t exp = pow(2.0, -currentExp);
      |        if(fractional >= exp) {
      |            fractional = fractional - exp;
      |            result = result | ((__int128)1 << (fractionBits - currentExp));
      |        }
      |    }
      |    return result & bitmask(integerBits + fractionBits);
      |}
      |
      |inline void printBinary(__int128 x) {
      |    std::cout << std::bitset<128>(x);
      |}
      |
      |inline void printlnBinary(__int128 x) {
      |    printBinary(x);
      |    std::cout << std::endl;
      |}
      |
      |class InterpolationHelper{
      |public:
      |    InterpolationHelper(uint32_t integer, uint32_t fraction, double_t error) : maxError(error), integer(integer), fraction(fraction){};
      |
      |    double_t maxError = pow(10.0, -6.0), initialStep =            1.0;
      |    uint32_t integer, fraction;
      |    double_t f(double_t x) { return log(1.0 + pow(2.0, x)) / log(2.0); }
      |    double_t finv(double_t y) { return log(pow(2.0, y) - 1.0) / log(2.0); }
      |    double_t fd1(double_t x) { return pow(2.0, x) / (1.0 + pow(2.0, x)); }
      |    double_t fd3(double_t x) { return -(pow(2.0, x) * pow(2.0, x) - 1.0) * pow(log(2.0), 2.0) / pow(1.0 + pow(2.0, x), 3.0); }
      |    double_t snap2Lower(double_t x, double_t h) { return floor(x / h) * h; }
      |    double_t err_lu(double_t l, double_t u, double_t h) {
      |        if(u < xfd3max) {
      |            return 1.0 / (9.0 *sqrt(3.0)) * pow(h,3.0) * fabs(fd3(u));
      |        } else if (l > xfd3max) {
      |            return 1.0 / (9.0 * sqrt(3.0)) * pow(h, 3.0) * fabs(fd3(l));
      |        } else {
      |            return 1.0 / (9.0 * sqrt(3.0)) * pow(h, 3.0) * fabs(fd3(xfd3max));
      |        }
      |    }
      |
      |    double_t xfd3max =                          log(2.0 - sqrt(3.0)) / log(2.0);
      |    std::pair<double_t, double_t > pStart =     std::make_pair(finv(maxError), maxError);
      |    std::vector<std::tuple<double_t , double_t , double_t >> pieces = getPieces();
      |    QuadSpline initialSpline =                  getInitialSpline();
      |    std::vector<double_t > range =              getRange();
      |    std::vector<QuadSpline> splines =           getSplines();
      |    std::tuple<uint32_t , uint32_t , uint32_t > addressing = getAddressing();
      |    std::vector<std::pair<uint32_t ,QuadSpline>> lookup = fillRange();
      |
      |    std::vector<std::tuple<double_t , double_t , double_t >> getPieces(){
      |        std::vector<std::tuple<double_t, double_t, double_t>> p;
      |        double_t h_curr = initialStep;
      |        double_t x_curr = snap2Lower(pStart.first, h_curr);
      |        double_t x_next = x_curr;
      |        while(x_curr < 0.0) {
      |            if (err_lu(x_curr, x_next + h_curr, h_curr) <= maxError && x_next < 0.0) {
      |                x_next = x_next + h_curr;
      |            } else {
      |                p.emplace_back(std::make_tuple(x_curr, x_next, h_curr));
      |                x_curr = x_next;
      |                h_curr = h_curr / 2.0;
      |            }
      |        }
      |        return p;
      |    };
      |
      |    QuadSpline getInitialSpline() {
      |        std::tuple<double_t, double_t, double_t > head(pieces[0]);
      |        double_t x = std::get<0>(head);
      |        double_t h = std::get<2>(head);
      |        return {x, f(x), 0.0, fd1(x), f(x), h};
      |    }
      |
      |    std::vector<double_t > getRange() {
      |        std::vector<double_t > r;
      |        for(std::tuple<double_t , double_t , double_t > piece : pieces) {
      |            for(double_t i = std::get<0>(piece); i < std::get<1>(piece); i = i + std::get<2>(piece)) {
      |                r.emplace_back(i);
      |            }
      |        }
      |        r.emplace_back(0.0);
      |        return r;
      |    }
      |
      |    std::vector<QuadSpline> getSplines() {
      |        std::vector<QuadSpline> s;
      |        s.emplace_back(initialSpline);
      |        std::vector<std::pair<double_t , double_t >> points;
      |        for(double_t x : range)  {
      |            double_t y = f(x);
      |            points.emplace_back(std::make_pair(x, y));
      |        }
      |
      |        for(size_t i = 0; i < points.size() - 1; i ++) {
      |            auto pn = points[i];
      |            auto pn1 = points[i + 1];
      |            auto sn = s[i];
      |            auto sn1 = generateSpline(sn, pn, pn1);
      |            s.emplace_back(sn1);
      |        }
      |        return s;
      |
      |    }
      |
      |    static QuadSpline generateSpline(QuadSpline &sn, std::pair<double_t, double_t> pn1, std::pair<double_t, double_t> pn2) {
      |        double_t a0 = sn.a, b0 = sn.b, x1 = std::get<0>(pn1), x2 = std::get<0>(pn2), y1 = std::get<1>(pn1), y2= std::get<1>(pn2);
      |
      |        double_t a1 = (2.0 * a0 * x1 * (x1 - x2) + b0 * (x1 - x2) - y1 + y2) / pow(x1 - x2, 2.0);
      |        double_t b1 = (2.0 * x1 * (a0 * (x2 * x2 - x1 * x1) + y1 - y2) + b0 * (x2 * x2 - x1 * x1)) / pow(x1 - x2, 2.0);
      |        double_t c1 = (2.0 * a0 * x1 * x1 * x2 * (x1 - x2) + b0 * x1 * x2 * (x1 - x2) + x1 * x1 * y2 - 2.0 * x1 * x2 * y1 + x2 * x2 * y1) / pow(x1 - x2, 2.0);
      |
      |        return {x2, y2, a1, b1, c1, x2 - x1};
      |    }
      |
      |    QuadSpline getSpline(double_t x) {
      |        for(auto spline : splines) {
      |            if(x<= spline.x) {
      |                return spline;
      |            }
      |        }
      |        return splines[0];
      |    }
      |
      |    std::tuple<uint32_t, uint32_t, uint32_t > getAddressing() {
      |        double_t stepwidth = splines[splines.size() - 1].h;
      |        double_t xDist = splines[splines.size() - 1].x - splines[0].x;
      |        double_t addresses = xDist / stepwidth;
      |
      |        uint32_t addressBits = ceil(log(addresses) / log(2.0));
      |        uint32_t lowestBit = ceil(log(splines[splines.size() - 1].h) / log(2.0));
      |        uint32_t highestBit = ceil(log(fabs(splines[0].x)) / log(2.0));
      |        return {highestBit + fraction - 1, fraction + lowestBit, addressBits};
      |    }
      |
      |    std::vector<std::pair<uint32_t ,QuadSpline>> fillRange() {
      |        std::vector<std::pair<uint32_t ,QuadSpline>> fullLookupTab;
      |        double_t xStart = 0.0;
      |        double_t xEnd = splines[0].x;
      |        double_t step = splines[splines.size() - 1].h;
      |        uint32_t counter = 0;
      |        for(auto i = xStart; i >= xEnd; i = i - step) {
      |            fullLookupTab.emplace_back(std::make_pair(counter, getSpline(i)));
      |            counter ++;
      |        }
      |        return fullLookupTab;
      |    }
      |
      |    Interpolator* getInterpolator() {
      |        std::vector<__int128> a,b,c;
      |        for(auto spline : lookup) {
      |            a.emplace_back(value2fixedInt(spline.second.a, integer, fraction));
      |            b.emplace_back(value2fixedInt(spline.second.b, integer, fraction));
      |            c.emplace_back(value2fixedInt(spline.second.c, integer, fraction));
      |        }
      |
      |        auto lastInterpolated = value2fixedInt(snap2Lower(pStart.first, initialStep), integer, fraction);
      |        return new Interpolator{integer, fraction, std::move(a),std::move(b),std::move(c),std::get<0>(addressing),std::get<1>(addressing),std::get<2>(addressing), lastInterpolated};
      |    }
      |
      |};
      |
      |static void initializeInterpolator(int integer, int fraction, double error) {
      |    interpolator = InterpolationHelper(integer, fraction, error).getInterpolator();
      |}
      |
      |template<int integer, int fraction>
      |struct LNS {
      |private:
      |    bool sign, zero;
      |    __int128 exponent;
      |
      |public:
      |
      |    LNS operator+(LNS other){
      |        if(zero) {
      |            return LNS(other.exponent, other.sign, other.zero);
      |        }
      |        if(other.zero) {
      |            return LNS(exponent, sign, zero);
      |        }
      |
      |        bool magnitudeGE = exponent >= other.exponent;
      |        bool aGE = other.zero || (other.sign && !sign) || (sign && other.sign && !magnitudeGE);
      |
      |        __int128 diff;
      |        if(aGE) {
      |            diff = other.exponent - exponent;
      |        } else {
      |            diff = exponent - other.exponent;
      |        }
      |
      |
      |        __int128 y = interpolator->ip(diff);
      |
      |        __int128 exp;
      |        if(aGE) {
      |            exp = (exponent - y) & bitmask(integer + fraction);
      |        } else {
      |            exp = (other.exponent - y) & bitmask(integer + fraction);
      |        }
      |        bool rsign = true;
      |        if(exp == 0) {
      |            rsign = false;
      |        }
      |        return LNS<integer, fraction>(exp, rsign, false);
      |    }
      |
      |    LNS operator*(LNS other) {
      |        if(zero || other.zero) { // 0*0 = 0
      |            return LNS<integer, fraction>(0, false, true);
      |        } else if(!sign) { // 0 * x= 0
      |            return LNS<integer, fraction>(other.exponent, other.sign, other.zero);
      |        } else if(!other.sign) { // x * 0 = 0
      |            return LNS<integer, fraction>(exponent, sign, zero);
      |        } else { // x * y...
      |            __int128 resultExp = (exponent + other.exponent) & bitmask(integer + fraction);
      |            if(checkOverflow(exponent + other.exponent, integer + fraction)) {
      |                return LNS<integer, fraction>(0, false, true);
      |            }
      |            return LNS<integer, fraction>(resultExp, sign, false);
      |        }
      |    }
      |
      |    LNS(double value){
      |        if(value <= 0.0) {
      |            zero = true;
      |            sign = false;
      |            exponent = 0;
      |        } else if(value >= 1.0) {
      |            zero = false;
      |            sign = false;
      |            exponent = 0;
      |        } else {
      |            double_t logval = log(value) / log(2.0);
      |            zero = false;
      |            sign = true;
      |            auto exp =  value2fixedInt(logval, integer, fraction);
      |            if(checkOverflow(exp, integer + fraction)) {
      |                sign = false;
      |                zero = true;
      |                exponent = 0;
      |            } else {
      |                exponent = exp;
      |            }
      |        }
      |    }
      |
      |    LNS(__int128 exponent, bool sign, bool zero) : exponent(exponent), sign(sign), zero(zero) {};
      |
      |    void println() {
      |        std::cout <<"Q" << integer << "." << fraction << "\tzero: " << zero << " sign: " << sign << " exp: " << std::bitset<integer + fraction>(exponent) <<
      |                " (Exp-Value: " << exponentAsDouble() << ", Value: "<< getEncodedValue() <<")" << std::endl;
      |    }
      |
      |    double_t exponentAsDouble() {
      |        std::bitset<integer + fraction> bit(exponent);
      |        double_t result = 0.0;
      |        for(size_t i = 0; i < integer + fraction; i++) {
      |            int32_t exp = i - fraction;
      |            double_t val = pow(2.0, exp);
      |            if(bit[i]) {
      |                result += val;
      |            }
      |        }
      |        if(sign) return -result;
      |        else return result;
      |    }
      |
      |    double_t getEncodedValue() {
      |        if(zero) return 0.0;
      |        else if(!sign) return 1.0;
      |        else return pow(2.0, exponentAsDouble());
      |    }
      |
      |};
      |
      |typedef struct LNS<LNS_INTEGER_BITS, LNS_FRACTION_BITS> lns_t;
      |
      |inline double lns_get_value(lns_t& lns){
      |  return (double) lns.getEncodedValue();
      |}
      |
      |
      |#endif //LNSOFT_LNS_H
      |""".stripMargin
}
