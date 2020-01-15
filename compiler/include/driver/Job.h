//
// Created by ls on 1/14/20.
//

#ifndef SPNC_JOB_H
#define SPNC_JOB_H

#include <vector>
#include <memory>
#include "Actions.h"

namespace spnc {

    template<typename Output>
    class Job {

    public:

        ActionBase& addAction(std::unique_ptr<ActionBase> action){
          actions.push_back(std::move(action));
          return *actions.back();
        }

        ActionBase& setFinalAction(std::unique_ptr<ActionWithOutput<Output>> action){
          finalAction = action.get();
          actions.push_back(std::move(action));
          return *actions.back();
        }

        Output& execute(){
          return finalAction->execute();
        }

    private:

        std::vector<std::unique_ptr<ActionBase>> actions;

        ActionWithOutput<Output>* finalAction;

    };
}




#endif //SPNC_JOB_H
