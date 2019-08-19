#ifndef _SPN_COMPILER_RT_H
#define _SPN_COMPILER_RT_H

#include <limits>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

typedef std::numeric_limits<double> double_limits;

struct dynamic_range_t {double minimum_value; double maximum_value;};
typedef struct dynamic_range_t dynamic_range;
inline dynamic_range range{double_limits::max(), double_limits::lowest()};
inline std::vector<std::pair<unsigned int, double>> * values = new std::vector<std::pair<unsigned int, double>>{};

inline double register_range(unsigned int level, double value) noexcept {
	range.minimum_value = (value < range.minimum_value) ? value : range.minimum_value;
	range.maximum_value = (value > range.maximum_value) ? value : range.maximum_value;
	values->push_back({level, value});
	return value;
}

inline void report_range(){
  std::cout << std::setprecision(double_limits::max_digits10)
    << "Smallest value " << range.minimum_value
    << " Largest value " << range.maximum_value << std::endl;
  std::ofstream myfile;
  myfile.open ("values.txt");
  myfile << std::setprecision(double_limits::max_digits10);
  myfile << "level,value" << std::endl;
  for(auto v : *values){
    myfile << v.first << "," << v.second << std::endl;
  }
  myfile.close();
  delete values;
}
#endif
