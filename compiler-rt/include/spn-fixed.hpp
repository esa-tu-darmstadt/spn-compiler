//
// Created by lukas on 04.08.19.
//

#ifndef LNSOFT_FIXEDPOINT_H
#define LNSOFT_FIXEDPOINT_H

#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <bitset>

#ifndef SPN_FIXED_INTEGER_BITS
#define SPN_FIXED_INTEGER_BITS 8
#endif

#ifndef SPN_FIXED_FRACTION_BITS
#define SPN_FIXED_FRACTION_BITS 8
#endif

template <int integer, int fraction>
struct FixedPoint {
private:
    __int128 val;

    __int128 bitmask(uint32_t bits) {
        __int128 result = 0;
        for (uint32_t i = 0; i < bits; i++) {
            result = (result << 1) + 1;
        }
        return result;
    }

    __int128 value2fixedInt(double_t value, uint32_t integerBits, uint32_t fractionBits) {
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
    
    /**
     * Convert value to double.
     */
    double getEncodedValue() {
        std::bitset<integer + fraction> bit(val);
        double_t  res = 0.0;
        for(auto i = 0; i < integer + fraction; i++) {
            int32_t exp = i - fraction;
            double_t value = pow(2.0, exp);
            if(bit[i]) {
                res += value;
            }
        }
        return res;
    }

public:
    /**
     * Add both numbers and mask the potential overflow to keep the same encoding.
     */
    FixedPoint operator+(FixedPoint other) {
        __int128 newVal = val + other.val;
        return FixedPoint(newVal & bitmask(integer + fraction));
    }

    /**
     * Multiply both numbers and apply a corresponding rightshift and bitmask to keep the encoding.
     */
    FixedPoint operator*(FixedPoint other) {
        __int128 newVal = (val + other.val) >> fraction;
        return FixedPoint(newVal & bitmask(integer + fraction));
    }

    /**
     * Transform Double to Fixed Point in the corrsponding encoding.
     */
    FixedPoint(double dValue) {
        if(integer + fraction > 64) {
            std::cout << "Cannot instantiate Fixed Point numbers with more than 64 bits." << std::endl;
            exit (EXIT_FAILURE);
        }
        val = value2fixedInt(dValue, integer, fraction);
    }

    FixedPoint(__int128 value) : val(value) {}

    operator double() {
      return getEncodedValue();
    }

};

typedef struct FixedPoint<SPN_FIXED_INTEGER_BITS, SPN_FIXED_FRACTION_BITS> spn_fixed_t;

#endif //LNSOFT_FIXEDPOINT_H
