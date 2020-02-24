//
// Created by ls on 1/14/20.
//

#ifndef SPNC_JOB_H
#define SPNC_JOB_H

#include <vector>
#include <memory>
#include <type_traits>
#include "Actions.h"

namespace spnc {

    template<typename Output>
    class Job {

    public:

      template<typename A, typename ...T>
      A& insertAction(T&& ... args) {
        static_assert(std::is_base_of<ActionBase, A>::value, "Must be an action derived from ActionBase!");
        actions.push_back(std::make_unique<A>(std::forward<T>(args)...));
        return *((A*) actions.back().get());
      }

      template<typename A, typename ...T>
      A& insertFinalAction(T&& ... args) {
        static_assert(std::is_base_of<ActionWithOutput<Output>, A>::value, "Must be an action with correct output!");
        auto a = std::make_unique<A>(std::forward<T>(args)...);
        finalAction = a.get();
        actions.push_back(std::move(a));
        return *((A*) actions.back().get());
      }

      ActionBase& addAction(std::unique_ptr<ActionBase> action) {
        actions.push_back(std::move(action));
        return *actions.back();
      }

      ActionBase& setFinalAction(std::unique_ptr<ActionWithOutput < Output>>
      action) {
        finalAction = action.get();
        actions.push_back(std::move(action));
        return *actions.back();
      }

      Output& execute() {
        return finalAction->execute();
      }

    private:

        std::vector<std::unique_ptr<ActionBase>> actions;

        ActionWithOutput<Output>* finalAction;

    };
}




#endif //SPNC_JOB_H
