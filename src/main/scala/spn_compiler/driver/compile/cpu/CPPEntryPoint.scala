package spn_compiler.driver.compile.cpu

import java.io.{BufferedWriter, File, FileWriter}

object CPPEntryPoint {

  def writeMain(mainFile : File) : Unit = {
    val writer = new BufferedWriter(new FileWriter(mainFile))
    writer.write(mainCode)
    writer.close()
  }

  private val mainCode =
    """#include <iostream>
      |#include <fstream>
      |#include <sstream>
      |#include <iomanip>
      |#include <cmath>
      |#include <chrono>
      |#include <vector>
      |#include "spn.hpp"
      |
      |int* readInputSamples(char * inputfile, int * sample_count){
      |    std::ifstream infile(inputfile);
      |    std::string line;
      |    std::vector<std::string> lines;
      |    while(std::getline(infile, line)){
      |        lines.push_back(line);
      |    }
      |
      |    std::vector<int> * input_data = new std::vector<int>();
      |    int sample_count_int = 0;
      |    for(const std::string& s : lines){
      |        std::istringstream stream(s);
      |        std::string token;
      |        int value_count = 0;
      |        while(std::getline(stream, token, ';')){
      |            std::istringstream value_string(token);
      |            double value;
      |            if(!(value_string >> value)){
      |                std::cout << "ERROR: Could not parse double from " << token.c_str() << std::endl;
      |                exit(-1);
      |            }
      |            input_data->push_back((int) value);
      |            ++value_count;
      |        }
      |        ++sample_count_int;
      |    }
      |
      |    std::cout << "Read " << sample_count_int << " input samples" << std::endl;
      |    *sample_count = sample_count_int;
      |    return input_data->data();
      |}
      |
      |double* readReferenceValues(char * outputfile, int sample_count){
      |    std::ifstream infile(outputfile);
      |    std::string line;
      |    auto * output_data = (double*) malloc(sample_count * sizeof(double));
      |    int count = 0;
      |    while(count < sample_count && std::getline(infile, line)){
      |        std::istringstream value_string(line);
      |        double value;
      |        if(!(value_string >> value)){
      |            std::cout << "ERROR: Could not parse double from " << line.c_str() << std::endl;
      |            exit(-1);
      |        }
      |        output_data[count] = value;
      |        ++count;
      |    }
      |
      |    std::cout << "Read " << count << " reference values" << std::endl;
      |    return output_data;
      |}
      |
      |int main(int argc, char ** argv) {
      |    if(argc < 2 || argc > 3 ){
      |        std::cout << "Please provide input-file as first argument and - optionally- output-data file as second argument!" << std::endl;
      |	exit(-1);
      |    }
      |    bool has_reference = (argc==3);
      |    int sample_count = 0;
      |
      |    void * input_data = readInputSamples(argv[1], &sample_count);
      |    double result[sample_count];
      |    for(int i=0; i<sample_count; ++i){
      |        result[i] = 42.0;
      |    }
      |
      |    auto begin = std::chrono::high_resolution_clock::now();
      |
      |    // TODO Kernel invocation
      |    spn_toplevel(sample_count, (activation_t*) input_data, result);
      |
      |    auto end = std::chrono::high_resolution_clock::now();
      |    if(has_reference){
      |    	double * reference_data = readReferenceValues(argv[2], sample_count);
      |    	int num_errors = 0;
      |    	std::cout << "Sample count: " << sample_count << std::endl;
      |    	for(int i=0; i<sample_count; ++i){
      |            if(std::abs(std::log(result[i])-reference_data[i])>1e-6){
      |            	std::cout << "ERROR: Significant deviation @" << i << ": " << std::log(result[i]) << " (" << result[i] << ") " << " vs. " << reference_data[i] << std::endl;
      |            	++num_errors;
      |            }
      |    	}
      |    	if(num_errors==0){
      |            std::cout << "COMPUTATION OK" << std::endl;
      |    	}
      |    }
      |
      |    std::cout << std::setprecision(15)<< "time per instance " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / (double) sample_count << " us" << std::endl;
      |    std::cout << std::setprecision(15) << "time per task " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()  << " us" << std::endl;
      |
      |    return 0;
      |}
      |""".stripMargin
}
