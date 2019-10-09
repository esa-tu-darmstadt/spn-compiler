#include <stdio.h>
#include <cmath>

template<typename T>
struct softLNS{
public:

  explicit softLNS(){
    value = float(0);
    isZero = false;
	}

	softLNS(T t){
    if (t == 0) {
      value = T(0.0);
      isZero = true;
    } else {
      value = T( log(static_cast<T>(t)));
      isZero = false;
    }
	}

  explicit softLNS(float t, bool z) : value(t), isZero(z) {}
  explicit softLNS(float t, bool z, bool init) : value(log(t)), isZero(z) {}
  explicit softLNS(double t, bool z) : value(t), isZero(z) {}
  explicit softLNS(double t, bool z, bool init) : value(log(t)), isZero(z) {}

	friend softLNS operator+(const softLNS & a, const softLNS & b){
    if (a.isZero){
      return softLNS{((T) b.value), b.isZero};
    } else if (b.isZero){
      return softLNS{((T) a.value), a.isZero};
    } else if (a.value > b.value){
      return softLNS{((T) a.value) + log(1 + exp(((T) b.value) - ((T) a.value))), false};
    } else {
      return softLNS{((T) b.value) + log(1 + exp(((T) a.value) - ((T) b.value))), false};
    }
	}

	friend softLNS operator*(const softLNS & a, const softLNS & b){
    if (a.isZero || b.isZero){
      return softLNS{T(0), true};
    } else {
      return softLNS{((T) a.value) + ((T) b.value), false};
    }
	}

	operator double(){
    if (!isZero) {
      return exp(static_cast<double>(value));
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

typedef struct softLNS<_Float16> spn_lns_h;
typedef struct softLNS<float> spn_lns_f;
typedef struct softLNS<double> spn_lns_d;
