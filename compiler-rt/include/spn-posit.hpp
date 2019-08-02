#ifndef SPN_POSIT_H
#define SPN_POSIT_H

#include "posit/lib/posit.h"

#ifndef POSIT_SIZE_N
  #define POSIT_SIZE_N 32
#endif

#ifndef POSIT_SIZE_ES
  #define POSIT_SIZE_ES 6
#endif

template<int N, int ES>
struct softposit{
  
  softposit(double d) : posit{N, ES}{
    posit.set(d);
  }
  
  softposit(Posit p) : posit{N, ES}{
    posit.set(p);
  }
  
  softposit(softposit & s) : posit{N, ES}{
    posit.set(s.posit);
  }
  
  softposit& operator=(const softposit& s){
    posit.set(s.posit);
  }
  
  softposit(softposit && s) : posit{N, ES}{
    posit.set(s.posit);
  }
  
  softposit& operator=(softposit&& s){
    posit.set(s.posit);
  }
  
  friend softposit operator+(const softposit & a, const softposit & b){
    return softposit{(a.posit + b.posit)};
  }
  
  friend softposit operator*(const softposit & a, const softposit & b){
    return softposit{(a.posit * b.posit)};
  }
  
  operator double() {
    return posit.getDouble();
  }
  
private:
  Posit posit;
};

typedef struct softposit<POSIT_SIZE_N, POSIT_SIZE_ES> posit_t;

#endif
