package spn_compiler.driver.compile.cpu

import java.io.{BufferedWriter, File, FileWriter}

object CPPEntryPoint {

  def writeMain(mainFile : File) : Unit = {
    val writer = new BufferedWriter(new FileWriter(mainFile))
    writer.write(mainCode)
    writer.close()
  }

  private val mainCode =
    """#include <vector>
      |#include <fstream>
      |#include <iostream>
      |#include <sstream>
      |#include <string>
      |#include <cmath>
      |#include <limits>
      |#include <iomanip>
      |#include <chrono>
      |#include "spn.hpp"
      |#ifdef SPN_PROFILE
      |#include "spn-compiler-rt.hpp"
      |#endif
      |
      |typedef std::numeric_limits<double> double_limits;
      |
      |std::vector<int>* readInputData(std::string inputfile, int * num_samples){
      |  std::vector<int> * inputdata = new std::vector<int>{};
      |  std::ifstream infile;
      |  infile.open(inputfile);
      |  if(!infile.is_open()){
      |    std::cerr << "Could not open input file!" << std::endl;
      |    exit(-1);
      |  }
      |  std::string line;
      |  int samples = 0;
      |  while(std::getline(infile, line)){
      |    std::istringstream linestream(line);
      |    std::string token;
      |    while(std::getline(linestream, token, ';')){
      |        std::istringstream value_string(token);
      |        double value;
      |        if(!(value_string >> value)){
      |            std::cout << "ERROR: Could not parse double from "
      |              << token.c_str() << std::endl;
      |            exit(-1);
      |        }
      |        inputdata->push_back((int) value);
      |    }
      |    samples++;
      |  }
      |  *num_samples = samples;
      |  return inputdata;
      |}
      |
      |std::vector<double>* readReferenceData(std::string referencefile){
      |  std::vector<double> * referenceData = new std::vector<double>{};
      |  std::ifstream reffile;
      |  reffile.open(referencefile);
      |  if(!reffile.is_open()){
      |    std::cerr << "Could not open reference file!" << std::endl;
      |    exit(-1);
      |  }
      |  std::string line;
      |  while(std::getline(reffile, line)){
      |    std::istringstream value_string(line);
      |    double value;
      |    if(!(value_string >> value)){
      |      std::cerr << "ERROR: Could not parse double from "
      |        << line.c_str() << std::endl;
      |      exit(-1);
      |    }
      |    referenceData->push_back(value);
      |  }
      |  return referenceData;
      |}
      |
      |int main(int argc, char ** argv) {
      |  if(argc < 2){
      |    std::cout << "Usage: spn.out $INPUT_DATA_FILE [$REFERENCE_DATA_FILE]" << std::endl;
      |    exit(-1);
      |  }
      |  int num_samples;
      |  std::vector<int> * inputdata = readInputData(std::string{argv[1]}, &num_samples);
      |  std::vector<double> * results = new std::vector<double>(num_samples, 42.0);
      |  auto begin = std::chrono::high_resolution_clock::now();
      |  spn_toplevel(num_samples, (activation_t*) inputdata->data(), results->data());
      |  auto end = std::chrono::high_resolution_clock::now();
      |  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin);
      |  int num_errors{0};
      |  double max_error{0.0};
      |  double min_error{1.0};
      |  double avg_error{0.0};
      |  if(argc > 2){
      |    std::cout << "Comparing computation results against reference data...\n";
      |    std::vector<double> * referenceData = readReferenceData(std::string{argv[2]});
      |    for(int i=0; i < referenceData->size(); ++i){
      |      double golden = exp((*referenceData)[i]);
      |      double error = fabs(golden - (*results)[i]);
      |      max_error = std::max(max_error, error);
      |      min_error = std::min(min_error, error);
      |      avg_error += error;
      |      if(error > 1E-6){
      |        std::cerr << "Significant result deviation " << golden
      |          << " vs. " << (*results)[i] << " error: " << error << std::endl;
      |        ++num_errors;
      |      }
      |    }
      |    avg_error /= referenceData->size();
      |    std::cout << std::setprecision(double_limits::max_digits10);
      |    std::cout << "max-error: " << max_error << std::endl;
      |    std::cout << "min-error: " << min_error << std::endl;
      |    std::cout << "avg-error: " << avg_error << std::endl;
      |    delete referenceData;
      |  }
      |
      |  if(num_errors == 0){
      |    std::cout << "Successfully completed computation" << std::endl;
      |  } else {
      |    std::cout << "Total number of computation errors: " << num_errors << std::endl;
      |  }
      |  std::cout << std::setprecision(15)<< "time per instance " << (duration.count() / (double) num_samples) << " us" << std::endl;
      |  std::cout << std::setprecision(15) << "time per task " << duration.count()  << " us" << std::endl;
      |  #ifdef SPN_PROFILE
      |  report_range();
      |  #endif
      |  delete inputdata;
      |  delete results;
      |
      |  return 0;
      |}
      |""".stripMargin
}
