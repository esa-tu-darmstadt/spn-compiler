#include <stdio.h>
#include <cmath>

template<typename T>
struct softLNS{
public:

  explicit softLNS(){
    value = float(0);
    isZero = false;
	}

	softLNS(float f){
    if (f == 0) {
      value = float(0);
      isZero = true;
    } else {
      value = log(f);
      isZero = false;
    }
	}

	softLNS(double d){
    if (d == 0) {
      value = double(0);
      isZero = true;
    } else {
      value = log(d);
      isZero = false;
    }
	}

  explicit softLNS(float f, bool z) : value(f), isZero(z) {}
  explicit softLNS(double d, bool z) : value(d), isZero(z) {}

	friend softLNS operator+(const softLNS & a, const softLNS & b){
    if (a.isZero){
      return softLNS{b.value, b.isZero};
    } else if (b.isZero){
      return softLNS{a.value, a.isZero};
    } else if (a.value > b.value){
      return softLNS{a.value + log(1 + exp(b.value - a.value)), false};
    } else {
      return softLNS{b.value + log(1 + exp(a.value - b.value)), false};
    }
	}

	friend softLNS operator*(const softLNS & a, const softLNS & b){
    if (a.isZero || b.isZero){
      return softLNS{T(0), true};
    } else {
      return softLNS{a.value + b.value, false};
    }
	}

	operator double(){
    if (!isZero) {
      return exp(value);
    }
		return 0.0;
	}

	void print(){
		printf("LNS {value: %f; isZero: %d}", value, isZero);
	}

private:
	T value;
  bool isZero;
};

typedef struct softLNS<float> spn_lns_f;
typedef struct softLNS<double> spn_lns_d;
