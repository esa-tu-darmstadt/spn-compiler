#include "TargetExecutionModel.h"

spnc::TargetExecutionModel &spnc::getGenericTargetExecutionModel() {
    static TargetExecutionModel genericTargetExecutionModel;
    return genericTargetExecutionModel;
}