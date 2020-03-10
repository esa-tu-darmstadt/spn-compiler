//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_INCLUDE_DRIVER_OPTIONS_H
#define SPNC_COMPILER_INCLUDE_DRIVER_OPTIONS_H

#include <unordered_map>
#include <iostream>
#include <memory>
#include <llvm/ADT/Optional.h>
#include <algorithm>
#include <map>
#include <vector>

namespace spnc {

  namespace interface {

    class OptValue {
    public:
      virtual ~OptValue() = default;
    };

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

      public:

        static std::string toLowerCase(const std::string& s) {
          std::string result = s;
          std::transform(result.begin(), result.end(), result.begin(),
                         [](unsigned char c) { return std::tolower(c); });
          return result;
        }

        template<typename Value>
        static Value parse(const std::string& value) {
          // As a default, try to construct a the value from a string.
          return Value{value};
        }

        template<>
        bool parse(const std::string& value) {
          std::string v = toLowerCase(value);
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

      OptValue& get(const std::string& key) const {
        return *config.at(key);
      }

      bool hasOption(const std::string& key) const {
        return config.count(key);
      }

    private:

      std::unordered_map<std::string, std::unique_ptr<OptValue>> config;

    };

    class Opt {

    public:

      explicit Opt(std::string key) : keyName{std::move(key)} {}

      virtual llvm::Optional<std::unique_ptr<OptValue>> parse(const std::string& key, const std::string& value) = 0;

      bool isPresent(const Configuration& config) {
        std::cout << "Key: " << keyName << std::endl;
        return config.hasOption(keyName);
      }

    public:

      std::string keyName;

    };

    class OptModifier {

    public:

      void initialize(Opt* opt) {
        modified = opt;
      }

      virtual bool verify(const Configuration& config) = 0;

      virtual ~OptModifier() = default;

    protected:

      Opt* modified = nullptr;

    };

    class Options {

    public:

      static void addOption(const std::string& key, Opt* opt,
                            std::initializer_list<OptModifier*> mods) {
        options.emplace(key, opt);
        for (auto m : mods) {
          m->initialize(opt);
          activeModifiers.push_back(m);
        }
      }

      static void dump() {
        std::cout << "Available options: " << std::endl;
        for (auto& k : options) {
          std::cout << k.first << std::endl;
        }
      }

      static std::unique_ptr<Configuration> parse(const std::map<std::string, std::string>& input) {
        auto config = std::make_unique<Configuration>();
        for (auto& o : input) {
          auto key = o.first;
          auto value = o.second;
          if (!options.count(key)) {
            throw std::runtime_error("Unknown compile option!");
          }
          auto parser = options.at(key);
          auto parseResult = parser->parse(key, value);
          if (!parseResult) {
            std::cout << "Could not parse option value " << value << " for " << key << std::endl;
          }
          config->push_back(key, std::move(parseResult.getValue()));
        }
        bool verified = true;
        for (auto m : activeModifiers) {
          verified &= m->verify(*config);
        }
        if (!verified) {
          throw std::runtime_error("Could not verify configuration constraints!");
        }
        return config;
      }

      template<class Modifier>
      static Modifier* registerModifier(std::unique_ptr<Modifier> mod) {
        Modifier* ptr = mod.get();
        allModifiers.push_back(std::move(mod));
        return ptr;
      }

      // DANGER ZONE: Correct static initialization order required.
      static std::unordered_map<std::string, Opt*> options;

      static std::vector<std::unique_ptr<OptModifier>> allModifiers;

      static std::vector<OptModifier*> activeModifiers;

    };

    template<typename Value>
    class Option : public Opt {

    public:

      explicit Option(std::string k, std::initializer_list<OptModifier*> modifier = {}) noexcept : Opt{std::move(k)} {
        Options::addOption(keyName, this, modifier);
      }

      Option(std::string k, Value defaultVal, std::initializer_list<OptModifier*> modifier = {}) noexcept : Opt{
          std::move(k)} {
        hasDefault = true;
        defaultValue = defaultVal;
        Options::addOption(keyName, this, modifier);
      }

      llvm::Optional<std::unique_ptr<OptValue>> parse(const std::string& key,
                                                      const std::string& value) override {
        if (key != keyName) {
          return llvm::None;
        }
        std::unique_ptr<OptValue>
            result = std::make_unique<OptionValue<Value>>(detail::OptionParsers::parse<Value>(value));
        return result;
      }

      Value get(const Configuration& config) {
        if (config.hasOption(keyName)) {
          return getVal(config);
        }
        if (hasDefault) {
          return defaultValue;
        }
        throw std::runtime_error("Trying to get value of non-present option!");
      }

      Value getOrElse(const Configuration& config, Value elseValue) {
        if (config.hasOption(keyName)) {
          return getVal(config);
        }
        return elseValue;
      }

    protected:

      Value getVal(const Configuration& config) {
        return dynamic_cast<OptionValue<Value>&>(config.get(keyName)).get();
      }

      bool hasDefault = false;

      Value defaultValue;

    };

    struct OptionEnumValue {
      std::string name;
      int value;
      std::string desc;
    };

    // Utility macros to easily define allowed values for an enum option.
#define EnumVal(ENUMVAL, DESC) \
      spnc::interface::OptionEnumValue {#ENUMVAL, int(ENUMVAL), DESC}

#define EnumValN(ENUMVAL, FLAGNAME, DESC) \
      spnc::interface::OptionEnumValue {FLAGNAME, int(ENUMVAL), DESC}

    class EnumOpt : public Option<int> {

    public:

      EnumOpt(std::string k, std::initializer_list<OptionEnumValue> options,
              std::initializer_list<OptModifier*> modifier = {}) : Option<int>{k, modifier} {
        for (auto& o : options) {
          enumValues.emplace(detail::OptionParsers::toLowerCase(o.name), o);
        }
      }

      template<typename D>
      EnumOpt(std::string k, D defaultVal, std::initializer_list<OptionEnumValue> options,
              std::initializer_list<OptModifier*> modifier = {}) : Option<int>{k, int(defaultVal), modifier} {
        for (auto& o : options) {
          enumValues.emplace(detail::OptionParsers::toLowerCase(o.name), o);
        }
      }

      llvm::Optional<std::unique_ptr<OptValue>> parse(const std::string& key,
                                                      const std::string& value) override {
        if (key != keyName) {
          // Key does not match this option.
          return llvm::None;
        }
        auto v = detail::OptionParsers::toLowerCase(value);
        if (!enumValues.count(v)) {
          // The value does not match any of the enum values.
          return llvm::None;
        }
        auto id = enumValues.at(v).value;
        std::unique_ptr<OptValue> result = std::make_unique<OptionValue<int>>(id);
        return result;
      }

    private:

      std::unordered_map<std::string, OptionEnumValue> enumValues;

    };

    class RequiredOpt : public OptModifier {

    public:

      bool verify(const Configuration& config) override {
        return modified->isPresent(config);
      }

    };

    static RequiredOpt* required() {
      return Options::registerModifier(std::make_unique<RequiredOpt>());
    }

    template<typename OptVal, typename Value>
    class Depends : public OptModifier {

    public:

      Depends(Option<OptVal>& opt, Value depend) : depOpt{opt}, depVal{depend} {
        static_assert(std::is_constructible<Value, OptVal>::value, "Incompatible type for dependency!");
      }

      bool verify(const Configuration& config) override {
        if (!depOpt.isPresent(config)) {
          return false;
        }
        OptVal val = depOpt.get(config);
        std::cout << "Dep: " << (Value{val} == depVal) << std::endl;
        return (Value{val} == depVal);
      }

    private:

      Option<OptVal>& depOpt;

      Value depVal;

    };

    template<typename OptVal, typename Value>
    static Depends<OptVal, Value>* depends(Option<OptVal>& opt, Value depend) {
      return Options::registerModifier(std::make_unique<Depends < OptVal, Value>>
      (opt, depend));
    }

  }

}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_OPTIONS_H
