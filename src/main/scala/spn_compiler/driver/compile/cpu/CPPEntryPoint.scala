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
      |#include <cstring>
      |#include <string>
      |#include <cmath>
      |#include <chrono>
      |#include <omp.h>
      |#include "spn.hpp"
      |#ifdef SPN_RANGE_PROFILE
      |#include "spn-compiler-rt.hpp"
      |#endif
      |#ifndef NUM_RUNS
      |#define NUM_RUNS 1
      |#endif
      |
      |#ifndef BLOCK_SIZE
      |  #define BLOCK_SIZE 100000
      |#endif
      |
      |#ifndef DATA_REPEAT
      |  #define DATA_REPEAT 1000
      |#endif
      |
      |#ifndef NUM_THREADS
      |  #define NUM_THREADS 8
      |#endif
      |
      |
      |std::vector<int>* readInputData(std::string inputfile, int * num_samples){
      |  std::vector<int> * inputdata = new std::vector<int>{};
      |  std::ifstream infile;
      |  infile.open(inputfile);
      |  if(!infile.is_open()){
      |    std::cerr << "Could not open input file!" << std::endl;
      |    exit(-1);
      |  }
      |  int sample_count = 0;
      |  std::string line;
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
      |    ++sample_count;
      |  }
      |  *num_samples = sample_count;
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
      |
      |int main(int argc, char ** argv) {
      |  if(argc < 2){
      |    std::cout << "Usage: tapasco-sw $INPUT_DATA_FILE [$REFERENCE_DATA_FILE]" << std::endl;
      |    exit(-1);
      |  }
      |  int input_samples;
      |  std::vector<int> * inputdata = readInputData(std::string{argv[1]}, &input_samples);
      |  activation_t * indata = new activation_t[input_samples*DATA_REPEAT];
      |  for(int i=0; i<DATA_REPEAT; ++i){
      |    std::memcpy(&(indata[i*input_samples]), inputdata->data(), input_samples*sizeof(activation_t));
      |  }
      |  auto num_samples = input_samples * DATA_REPEAT;
      |  int num_errors{0};
      |  std::chrono::milliseconds duration{0};
      |  for(int r=0; r < NUM_RUNS; ++r){
      |    std::vector<double> * results = new std::vector<double>(num_samples, 42.0);
      |    double * outdata = results->data();
      |
      |    int num_jobs = (int) ceil(((float) num_samples) / ((float) BLOCK_SIZE));
      |    auto begin = std::chrono::high_resolution_clock::now();
      |    #pragma omp parallel for num_threads(NUM_THREADS) firstprivate(indata, outdata)
      |    for(int i=0; i<num_jobs; ++i){
      |      int block_start = i * BLOCK_SIZE;
      |      int block_size = std::min((num_samples - block_start), BLOCK_SIZE);
      |      auto input_start = &(indata[block_start]);
      |      auto output_start = &(outdata[block_start]);
      |      spn_toplevel(block_size, input_start, output_start);
      |    }
      |    auto end = std::chrono::high_resolution_clock::now();
      |    duration = duration +  std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
      |    if(argc > 2){
      |      std::vector<double> * referenceData = readReferenceData(std::string{argv[2]});
      |      if(referenceData->size()!=input_samples){
      |        std::cout << "Not enough reference data!" << std::endl;
      |      }
      |      else{
      |        for(int i=0; i < num_samples; ++i){
      |          int reference_index = i % referenceData->size();
      |          double golden = (*referenceData)[reference_index];
      |          double fpga_result = log((*results)[i]);
      |          double error = fabs(golden - fpga_result);
      |          if(error > 1E-6){
      |            std::cerr << "Significant result deviation @" << i << ": " << golden
      |              << " vs. " << fpga_result << " error: " << error << std::endl;
      |            ++num_errors;
      |          }
      |        }
      |      }
      |      delete referenceData;
      |    }
      |    delete results;
      |  }
      |
      |  if(num_errors == 0){
      |    std::cout << "Successfully completed computation" << std::endl;
      |  } else {
      |    std::cout << "Total number of computation errors: " << num_errors << std::endl;
      |  }
      |  std::cout << "Computation took " << (((double) duration.count())/NUM_RUNS) << "ms" << std::endl;
      |  #ifdef SPN_RANGE_PROFILE
      |  report_range();
      |  #endif
      |  delete inputdata;
      |  delete[] indata;
      |
      |  return 0;
      |}
      |""".stripMargin
}
