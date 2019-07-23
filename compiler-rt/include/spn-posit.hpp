#ifndef SPN_POSIT_H
#define SPN_POSIT_H

#include "posit/posit.h"

#ifndef POSIT_SIZE_N
  #define POSIT_SIZE_N 32
#endif

#ifndef POSIT_SIZE_ES
  #define POSIT_SIZE_ES 6
#endif

#ifndef POSIT_STORAGE_TYPE
  #if POSIT_SIZE_N > 64
	  #pragma message "POSIT SIZE GREATER 64 BITS NOT SUPPORTED"
		#define POSIT_STORAGE_TYPE
	#elif POSIT_SIZE_N > 32
	  #define POSIT_STORAGE_TYPE int64_t
	#elif POSIT_SIZE_N > 16
	  #define POSIT_STORAGE_TYPE int32_t
	#elif POSIT_SIZE_N > 8
	  #define POSIT_STORAGE_TYPE int16_t
	#else
	  #define POSIT_STORAGE_TYPE int8_t
	#endif
#endif

#ifndef POSIT_FRACTION_TYPE
  #if POSIT_SIZE_N > 64
	  #pragma message "POSIT SIZE GREATER 64 BITS NOT SUPPORTED"
		#define POSIT_FRACTION_TYPE
	#elif POSIT_SIZE_N > 32
	  #define POSIT_FRACTION_TYPE uint64_t
	#elif POSIT_SIZE_N > 16
	  #define POSIT_FRACTION_TYPE uint64_t
	#elif POSIT_SIZE_N > 8
	  #define POSIT_FRACTION_TYPE uint32_t
	#else
	  #define POSIT_FRACTION_TYPE uint16_t
	#endif
#endif

typedef Posit<POSIT_STORAGE_TYPE, POSIT_SIZE_N, POSIT_SIZE_ES, 
  POSIT_FRACTION_TYPE, PositSpec::WithNan> posit_t;

#endif
