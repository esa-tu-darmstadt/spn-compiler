//
// Created by ls on 1/14/20.
//

#ifndef SPNC_ACTIONS_H
#define SPNC_ACTIONS_H

namespace spnc {

    class ActionBase {
    public:
        virtual ~ActionBase() = default;
    };

    template<typename Output>
    class ActionWithOutput : public ActionBase {
    public:
        virtual Output& execute() = 0;

        explicit ActionWithOutput() = default;

        ~ActionWithOutput() override = default;
    };

    template<typename Input, typename Output>
    class ActionSingleInput : public ActionWithOutput<Output> {

    public:
        explicit ActionSingleInput(ActionWithOutput<Input>& _input) : input{_input}{}

    protected:
        ActionWithOutput<Input>& input;

    };

    template<typename Input1, typename Input2, typename Output>
    class ActionDualInput : public ActionWithOutput<Output> {

    public:
        ActionDualInput(ActionWithOutput<Input1>& _input1, ActionWithOutput<Input2>& _input2)
          : input1{_input1}, input2{_input2} {}

    protected:
        ActionWithOutput<Input1>& input1;
        ActionWithOutput<Input2>& input2;

    };
}






#endif //SPNC_ACTIONS_H
