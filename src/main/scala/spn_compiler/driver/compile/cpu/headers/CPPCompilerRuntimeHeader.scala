package spn_compiler.driver.compile.cpu.headers

import java.io.{BufferedWriter, File, FileWriter}

object CPPCompilerRuntimeHeader {

  def writeHeader(file: File) : Unit = {
    val writer = new BufferedWriter(new FileWriter(file))
    writer.write(headerCode)
    writer.close()
  }

  private val headerCode =
    """#ifndef _SPN_COMPILER_RT_H
       |#define _SPN_COMPILER_RT_H
       |
       |#include <limits>
       |#include <iostream>
       |#include <fstream>
       |#include <iomanip>
       |#include <vector>
       |
       |typedef std::numeric_limits<double> double_limits;
       |
       |struct dynamic_range_t {double minimum_value; double maximum_value;};
       |typedef struct dynamic_range_t dynamic_range;
       |inline dynamic_range range{double_limits::max(), double_limits::lowest()};
       |inline std::vector<double> * values = new std::vector<double>{};
       |
       |inline double register_range(double value) noexcept {
       |	range.minimum_value = (value < range.minimum_value) ? value : range.minimum_value;
       |	range.maximum_value = (value > range.maximum_value) ? value : range.maximum_value;
       |	values->push_back(value);
       |	return value;
       |}
       |
       |inline void report_range(){
       |	std::cout << std::setprecision(double_limits::max_digits10)
       |		<< "Smallest value " << range.minimum_value
       |		<< " Largest value " << range.maximum_value << std::endl;
       |	std::ofstream myfile;
       |  myfile.open ("values.txt");
       |  myfile << std::setprecision(double_limits::max_digits10);
       |	for(auto v : *values){
       |		myfile << v << std::endl;
       |	}
       |  myfile.close();
       |	delete values;
       |}
       |#endif
       |""".stripMargin

}
