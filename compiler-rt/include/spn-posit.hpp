#ifndef SPN_POSIT_H
#define SPN_POSIT_H

#include <tuple>
#include <bitset>
#include <cmath>
#include <iostream>
#include <assert.h>

#ifndef POSIT_SIZE_N
  #define POSIT_SIZE_N 32
#endif

#ifndef POSIT_SIZE_ES
  #define POSIT_SIZE_ES 6
#endif

static bool debug = false;

inline __int128 intpow(__int128 x, __int128 y) {
    __int128 acc = 1;
    if(y < 0) {
        for(int i = 0; i > y; i--) {
            acc /= x;

        }
    }
    for(int i = 0; i < y; i++) {
        acc = acc * x;
    }
    return acc;
}

inline __int128 bitmask(uint32_t bits) {
    __int128 r = 0;
    for(auto i = 0; i < bits; i++) {
        r = (r << 1)+ 1;
    }
    return r;
}

inline __int128 value2fixedInt(double_t value, uint32_t integerBits, uint32_t fractionBits) {
    value = fabs(value);
    __int128 result = (__int128)floor(value) << fractionBits;
    double_t fractional = value - floor(value);

    for(int currentExp = 1; currentExp <= fractionBits; currentExp++) {
        double_t exp = pow(2.0, -currentExp);
        if(fractional >= exp) {
            fractional = fractional - exp;
            result = result | ((__int128)1 << (fractionBits - currentExp));
        }
    }
    return result & bitmask(integerBits + fractionBits);
}


template <int size, int es>
struct Posit {
private:
    bool sign = false, zero = false;
    int32_t regime, exponent;
    __int128 fraction;
    __int128 useed = intpow(2, intpow(2, es));

    uint32_t regimeLen, exponentLen, fractionLen;
    double _origin;
public:

    Posit(double  origin, int32_t reg, int32_t exp, __int128 frac) : _origin(origin), regime(reg), exponent(exp), fraction(frac) {
        regimeLen = regimeLenght();
        exponentLen = exponentLength();
        fractionLen = fractionLength();
        assert(exponent < pow(2.0, exponentLen + 1) );
        assert(regimeLen + exponentLen + fractionLen + 1 == size);
    }

    Posit(double value) {
        _origin = value;
        if (value == 0.0) {
            zero = true;
            regime = 0;
            regimeLen = size - 1;
            exponentLen = exponentLength();
            fractionLen = fractionLength();
        } else {
            regime = 0;
            exponent = 0;

            while(value < 1.0) {
                value *= useed;
                regime -= 1;
            }
            while(value >= 2.0) {
                value /= 2.0;
                exponent += 1;
            }
            regimeLen = regimeLenght();
            exponentLen = exponentLength();
            fractionLen = fractionLength();
            fraction = value2fixedInt(value - 1.0, 0, fractionLen);

        }

    }

    int32_t getFractionLength(__int128 frac)  {
        __int128 i = 0;
        int32_t len = 0;
        while(i < fraction) {
            i = (i << 1) | 1;
            len ++;
        }
        return len;
    }

    Posit operator*(Posit other) {
        __int128 one = 1;
        __int128 fracA = ((one << fractionLen) | fraction);
        __int128 fracB = ((one << other.fractionLen) | other.fraction);
        // Both Fractions are aligned with their decimal point in the same position
        int32_t expA = getFullExponent(), expB = other.getFullExponent();

        if(debug) {
            std::cout << std::endl;
            std::cout << std::bitset<64>(fracA).to_string() << " " << fractionLen << " " << expA << std::endl;
            std::cout << std::bitset<64>(fracB).to_string() << " " << other.fractionLen << " " << expB << std::endl;
        }
        __int128 fracR = (fracA * fracB);
        int32_t fracRLen = fractionLen + other.fractionLen;
        int32_t fullExpR = expA + expB;
        if(debug)
        std::cout << std::bitset<64>(fracR).to_string() << " " << fracRLen << " " << fullExpR << std::endl;

        int32_t regR = extractRegime(fullExpR);
        int32_t expR = extractExp(fullExpR, regR);
        if(debug)
        std::cout << regR << " " << expR << std::endl;

        int32_t regL = getregimeLenght(regR);
        int32_t expL = getexponentLength(expR, regL);
        int32_t fracL = size - 1 - regL - expL;
        fracR = fracR >> (fracRLen - fracL);
        if(debug)
        std::cout << std::bitset<64>(fracR).to_string() << " " << fracL << " " << fullExpR << std::endl;
        __int128 mask = bitmask(fracL + 1);
        if(fracR <= mask) {
            return Posit<size, es>(_origin * other._origin, regR, expR, fracR & bitmask(fracL));
        } else {
            fracR = fracR >> 1;
            expR = expR + 1 == intpow(2, es) ? 0 : expR + 1;
            if(expR == 0) { // Special Case with Regime Change;
                regR = regR + 1;
                fracL += 1;
                fracR = fracR << 1;
                return Posit<size, es>(_origin * other._origin, regR, expR, fracR & bitmask(fracL));
            } else {
                return Posit<size, es>(_origin * other._origin, regR, expR, fracR & bitmask(fracL));
            }

        }
    }

    Posit operator+(Posit other) {
        auto fracSize = fractionLen > other.fractionLen ? fractionLen : other.fractionLen;
        __int128 one = 1;
        __int128 fracA = ((one << fractionLen) | fraction) << (fracSize - fractionLen);
        __int128 fracB = ((one << other.fractionLen) | other.fraction) << (fracSize - other.fractionLen);
        // Both Fractions are aligned with their decimal point in the same position


        int32_t expA = getFullExponent(), expB = other.getFullExponent(), expR;
        __int128 fracR;
        if(expA == expB) {
            expR = expA + 1;
            fracR = (fracA + fracB) >> 1;
        } else if(expA < expB) {
            auto maxShiftless = intpow(2, other.fractionLen + 1) - 1;
            fracR = (fracB + (fracA >> (expB - expA)));
            expR = expB;
            if(fracR > maxShiftless) {
                fracR = fracR >> 1;
                expR = expR + 1;
            }
        } else {
            auto maxShiftless = intpow(2, fractionLen + 1) -1;
            fracR = (fracA + (fracB >> (expA - expB)));
            expR = expA;
            if(fracR > maxShiftless) {
                fracR = fracR >> 1;
                expR = expR + 1;
            }
        }
        // expR contains the full exp, fracR contains the correctly shifted Mantissa. So now we hide the implicit 1 and split the exp in runlegth and exp.
        int32_t reg = extractRegime(expR);
        int32_t exp = extractExp(expR, reg);
        __int128 mask = 1;
        while(mask < fracR) {
            mask = mask << 1;
        }
        mask = (mask - 1) >> 1;
        fracR = fracR & mask;

        return Posit(_origin + other._origin, reg, exp, fracR);
    }

    int32_t extractRegime(int32_t exp) {
        int32_t r = 0;
        while(exp < 0) {
            exp += intpow(2, es);
            r -= 1;
        }
        return r;
    }

    int32_t extractExp(int32_t exp, int32_t reg) {
        int32_t a = reg * intpow(2, es);
        int32_t e = 0;
        while(a + e != exp) e ++;
        return e;



    }

    int32_t getregimeLenght(int32_t reg) {
        if(reg == 0) return 2;
        else return -reg + 1;
    }

    int32_t regimeLenght() {
        return getregimeLenght(regime);
    }

    int32_t getexponentLength(int32_t exp, int32_t regLen) {
        if(size - 1 - regLen >= es) return es;
        else return size - 1 - regLen;
    }

    int32_t exponentLength() {
        return getexponentLength(exponent, regimeLen);
    }

    int32_t fractionLength() {
        if(size - 1 - regimeLen - exponentLen <= 0) return 0;
        else return size - 1 - regimeLen - exponentLen;
    }

    int32_t getFullExponent() {
        return regime * intpow(2, es) + exponent;
    }

    double getOrigin() {return _origin;}
    double getValue() {
        if(zero) return 0.0;
        else {
            auto scale = pow(2.0, regime * intpow(2, es) + exponent);
            auto mant = getMantissaValue();

            return scale * mant ;
        }
    }

    double getFractionValue() {
        double frac = static_cast<double>(fraction);
        double base = pow(2.0, static_cast<double>(fractionLen));
        return frac / base;
    }

    double getMantissaValue() { return 1.0 + getFractionValue(); }

    bool assertCorrectness() {
        auto up = _origin + 1e-6, low = _origin - 1e-6;
        return getValue() >= low && getValue() <= up;
    }

    std::bitset<size> regimeAsBitset() {
        if(regime == 0) {
            return std::bitset<size>(2);
        } else {
            return std::bitset<size>(1);
        }
    }

    std::string to_string() {
        std::bitset<size> result(fraction);
        std::bitset<size> exp(exponent);
        std::bitset<size> reg = regimeAsBitset();

        std::string x =  (result | (exp << fractionLen) | (reg << (fractionLen + exponentLen))).to_string();


        x.append(" ").append(std::to_string(regime)).append(" ").append(std::to_string(exponent)).append(" ").append(std::to_string(getMantissaValue()));
        return x;
    }

    operator double(){
      return getValue();
    }

};

typedef struct Posit<POSIT_SIZE_N, POSIT_SIZE_ES> posit_t;

#endif
