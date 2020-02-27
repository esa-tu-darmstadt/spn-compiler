#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

extern "C" {
  void spn_element(int32_t *in, double* out, int64_t sample_count);
}
int32_t* readInputSamples(char * inputfile, int64_t * sample_count, bool silent){
    std::ifstream infile(inputfile);
    std::string line;
    std::vector<std::string> lines;
    while(std::getline(infile, line)){
        lines.push_back(line);
    }

    uint64_t sample_width = std::count(lines[0].begin(), lines[0].end(), ';')+1;
    if (!silent)
      std::cout << "sample width " << sample_width << std::endl;
    auto * input_data = (int32_t*) malloc(sample_width * lines.size() * sizeof(int));
    int64_t sample_count_int = 0;
    for(const std::string& s : lines){
        std::istringstream stream(s);
        std::string token;
        int value_count = 0;
        while(std::getline(stream, token, ';')){
            std::istringstream value_string(token);
            double value;
            if(!(value_string >> value)){
                std::cout << "ERROR: Could not parse double from " << token.c_str() << std::endl;
                exit(-1);
            }
            input_data[sample_count_int*sample_width + value_count] = (int32_t) value;
            ++value_count;
        }
        ++sample_count_int;
    }
    if (!silent)
      std::cout << "Read " << sample_count_int << " input samples" << std::endl;
    *sample_count = sample_count_int;
    return input_data;
}

double* readReferenceValues(char * outputfile, int sample_count, bool silent){
    std::ifstream infile(outputfile);
    std::string line;
    auto * output_data = (double*) malloc(sample_count * sizeof(double));
    int count = 0;
    while(count < sample_count && std::getline(infile, line)){
        std::istringstream value_string(line);
        double value;
        if(!(value_string >> value)){
            std::cout << "ERROR: Could not parse double from " << line.c_str() << std::endl;
            exit(-1);
        }
        output_data[count] = value;
        ++count;
    }
    if (!silent)
      std::cout << "Read " << count << " reference values" << std::endl;
    return output_data;
}

int main(int argc, char ** argv) {
    if(argc < 3){
        std::cout << "Please provide input- and output-data file as first and second argument!" << std::endl;
	exit(1);
    }
    
    int64_t sample_count = 0;
    bool silent = (argc > 3);
    int32_t * input_data = readInputSamples(argv[1], &sample_count, silent);
    double result[sample_count];
    for(int i=0; i<sample_count; ++i){
        result[i] = 42.0;
    }
    double * reference_data;

    if (!silent)
      reference_data = readReferenceValues(argv[2], sample_count, silent);

    auto begin = std::chrono::high_resolution_clock::now();

    // TODO Kernel invocation
    spn_element(input_data, result, sample_count);

    auto end = std::chrono::high_resolution_clock::now();
    int num_errors = 0;
    if (!silent)
      std::cout << "Sample count: " << sample_count << std::endl;
    if (!silent) {
      for (int i = 0; i < sample_count; ++i) {
        if (std::abs(std::log(result[i]) - reference_data[i]) > 1e-6) {
          std::cout << "ERROR: Significant deviation @" << i << ": "
                    << std::log(result[i]) << " (" << result[i] << ") "
                    << " vs. " << reference_data[i] << std::endl;
          ++num_errors;
        }
      }
      if (num_errors == 0) {
        if (!silent)
          std::cout << "COMPUTATION OK" << std::endl;
      }
    }
    if (!silent) {
      std::cout << std::setprecision(15) << "time per instance "
                << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                         begin)
                           .count() /
                       (double)sample_count
                << " us" << std::endl;
      std::cout << std::setprecision(15) << "time per task "
                << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                         begin)
                       .count()
                << " us" << std::endl;
    } else {
      std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                         begin)
	.count() << std::endl;
    }

    return 0;
}
