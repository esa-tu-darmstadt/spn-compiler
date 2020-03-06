//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_INCLUDE_DRIVER_OPTIONS_H
#define SPNC_COMPILER_INCLUDE_DRIVER_OPTIONS_H

#include <unordered_map>
namespace spnc {

  class OptValue {};

  template<typename Value>
  class OptionValue : public OptValue {

  public:
    explicit OptionValue(Value v) : value{v} {}

    Value get() {
      return value;
    }

  private:
    Value value;

  };

  namespace detail {

    class OptionParsers {

      template<typename Value>
      static Value parse(const std::string& value) {
        // As a default, try to construct a the value from a string.
        return Value{value};
      }

      template<>
      bool parse(const std::string& value) {
        std::string v = value;
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return v == "true" || v == "yes";
      }

      template<>
      int parse(const std::string& value) {
        return std::stoi(value);
      }

      template<>
      double parse(const std::string& value) {
        return std::stod(value);
      }

      template<>
      std::string parse(const std::string& value) {
        return value;
      }

    };

  }

  class Configuration {

  public:

    void push_back(const std::string& key, std::unique_ptr<OptValue> value) {
      config.emplace(key, std::move(value));
    }

    OptValue& get(const std::string& key) {
      return *config[key];
    }

  private:

    std::unordered_map<std::string, std::unique_ptr<OptValue>> config;

  };

  class Opt {

  public:

    explicit Opt(std::string key) : keyName{std::move(key)} {}

    virtual llvm::Optional<OptValue> parse(const std::string& key, const std::string& value) = 0;

  protected:

    std::string keyName;

  };

  class Options {

  public:

    static void addOption(const std::string& key, Opt* opt) {
      options.emplace(key, opt);
    }

  private:

    static std::unordered_map<std::string, Opt*> options;

  };

  template<typename Value>
  class Option : public Opt {

  public:
    explicit Option(std::string k) : Opt{std::move(k)} {
      Options::addOption(keyName, this);
    }

    llvm::Optional<std::unique_ptr<OptionValue<Value>>> parse(const std::string& key,
                                                              const std::string& value) override {
      if (key != keyName) {
        return llvm::None;
      }
      return std::make_unique<OptionValue<Value>>(detail::OptionParsers::parse<Value>(value));
    }

  };

}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_OPTIONS_H
