#pragma once

#include "pipeline/PipelineStep.h"
#include "util/FileSystem.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "Kernel.h"


namespace spnc {

// NOTE: This can be extended to parse the sources to for example integrate them fully into
// the CIRCT ecosystem. This makes sense if the user wants full end-to-end optimization.
// However, for now we just load and paste files.
class HDLSourceInfo {
  std::vector<std::string> paths;
  std::vector<std::string> fileNames;
public:
};

class HDLSources {
  std::vector<std::string> sources;
public:
};

class LoadHDLSources : public StepBase, public StepWithResult<void> {
public:



  STEP_NAME("load-hdl-sources");
private:
  
};

}