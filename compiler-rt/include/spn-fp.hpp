#include <stdio.h>

#include <gmp.h>
#include <mpfr.h>

#ifndef SPN_FLOAT_MANTISSA
  #define SPN_FLOAT_MANTISSA 53
#endif

#ifndef SPN_E_MAX
  #define SPN_E_MAX 1023
#endif

#ifndef SPN_E_MIN
  #define SPN_E_MIN -1021
#endif

#ifndef SPN_RND
  #define SPN_RND GMP_RNDN
#endif

template<int mantissa, int exp_max, int exp_min>
struct softfloat{
public:
	
	explicit softfloat(){
		mpfr_set_emax(exp_max);
		mpfr_set_emin(exp_min);
		mpfr_init2(value, mantissa);
	}
	
	softfloat(double d){
		mpfr_set_emax(exp_max);
		mpfr_set_emin(exp_min);
		mpfr_init2(value, mantissa);
		mpfr_set_d(value, d, SPN_RND);
	}
	
	softfloat(mpfr_t & val){
		mpfr_set_emax(exp_max);
		mpfr_set_emin(exp_min);
		mpfr_init2(value, mantissa);
		mpfr_set(value, val, SPN_RND);
	}
	
	softfloat(softfloat & s){
		mpfr_set_emax(exp_max);
		mpfr_set_emin(exp_min);
		mpfr_init2(value, mantissa);
		mpfr_set(value, s.value, SPN_RND);
	}
	
	softfloat& operator=(const softfloat& s){
		mpfr_set_emax(exp_max);
		mpfr_set_emin(exp_min);
		mpfr_set(value, s.value, SPN_RND);
	}
	
	softfloat(softfloat && s){
		mpfr_set_emax(exp_max);
		mpfr_set_emin(exp_min);
		mpfr_init2(value, mantissa);
		mpfr_set(value, s.value, SPN_RND);
		mpfr_set_d(s.value, 0.0, SPN_RND);
	}
	
	softfloat& operator=(const softfloat&& s){
		mpfr_set_emax(exp_max);
		mpfr_set_emin(exp_min);
		mpfr_set(value, s.value, SPN_RND);
		mpfr_set_d(s.value, 0.0, SPN_RND);
	}
	
	~softfloat(){
		mpfr_clear(value);
	}
	
	friend softfloat operator+(const softfloat & a, const softfloat & b){
		mpfr_set_emax(exp_max);
		mpfr_set_emin(exp_min);
		mpfr_t sum;
		mpfr_init2(sum, mantissa);
		mpfr_add(sum, a.value, b.value, SPN_RND);
		softfloat ret = softfloat{sum};
		mpfr_clear(sum);
		return ret;
	}
	
	friend softfloat operator*(const softfloat & a, const softfloat & b){
		mpfr_set_emax(exp_max);
		mpfr_set_emin(exp_min);
		mpfr_t prod;
		mpfr_init2(prod, mantissa);
		mpfr_mul(prod, a.value, b.value, SPN_RND);
		mpfr_clear_underflow();
		softfloat ret = softfloat{prod};
		mpfr_clear(prod);
		return ret;
	}
	
	operator double() {
		return mpfr_get_d(value, SPN_RND);
	}
	
	void print() {
		mpfr_out_str (stdout, 10, 0, value, MPFR_RNDN);
	}
	
private:
	mpfr_t value;
};

typedef struct softfloat<SPN_FLOAT_MANTISSA, SPN_E_MAX, SPN_E_MIN> spn_float_t;
