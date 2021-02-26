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
#include <util/Logging.h>
#include <CLI11.hpp>

namespace spnc {

  ///
  /// Namespace for configuration interface components.
  ///
  namespace interface {

    ///
    /// Base class for the value of an option.
    ///
    class OptValue {
    public:
      virtual ~OptValue() = default;
    };

    /// Represents the value of an Option in the current Configuration.
    /// \tparam Value Type of the value stored in this Option.
    template<typename Value>
    class OptionValue : public OptValue {

    public:

      /// Constructor.
      /// \param v The value for this Option in the currenct Configuration.
      explicit OptionValue(Value v) : value{v} {}

      /// Get the values stored in this option.
      /// \return The value.
      Value get() {
        return value;
      }

    private:
      Value value;

    };

    namespace detail {

      ///
      /// Collection of parsers and helper methods for Options.
      ///
      class OptionParsers {

      public:

        /// Convert a string to all-lower-case letters.
        /// \param s String.
        /// \return A copy of the string with all letter converted to lower-case.
        static std::string toLowerCase(const std::string& s) {
          std::string result = s;
          std::transform(result.begin(), result.end(), result.begin(),
                         [](unsigned char c) { return std::tolower(c); });
          return result;
        }

        /// Parse a value from string.
        /// \tparam Value Type of the value to parse.
        /// \param value String containing the value.
        /// \return Parse result.
        template<typename Value>
        static Value parse(const std::string& value);

      };

    }

    ///
    /// Represents a concrete configuration, i.e., it stores OptionValues for all the options that were given.
    ///
    class Configuration {

    public:

      /// Push-back the OptionValue for the given key.
      /// \param key Identifier of the associated Option.
      /// \param value The concrete value of the Option for this Configuration.
      void push_back(const std::string& key, std::unique_ptr<OptValue> value) {
        config.emplace(key, std::move(value));
      }

      /// Get the concrete OptionValue for an Option.
      /// \param key Identifier of the Option.
      /// \return The value of the Option. Will throw if the option was not given in the configuration.
      OptValue& get(const std::string& key) const {
        return *config.at(key);
      }

      /// Check if an Option was specified in this Configuration.
      /// \param key Identifier of the Option.
      /// \return true if the Option was specified, false otherwise.
      bool hasOption(const std::string& key) const {
        return config.count(key);
      }

    private:

      std::unordered_map<std::string, std::unique_ptr<OptValue>> config;

    };

    ///
    /// Base class of all options.
    ///
    class Opt {

    public:

      /// Constructor.
      /// \param key Identifier of this option.
      explicit Opt(std::string key) : keyName{std::move(key)} {}

      /// Try to parse this option, if the key matches the identifier.
      /// \param key Identifier of the option to parse.
      /// \param value Specified value of the option.
      /// \return None if the key did not match the identifier, otherwise the OptionValue parsed from the string.
      virtual llvm::Optional<std::unique_ptr<OptValue>> parse(const std::string& key, const std::string& value) = 0;

      /// Check if this option was specified in the Configuration.
      /// \param config Configuration.
      /// \return true if an option with the matching identifier was present in the configuration.
      bool isPresent(const Configuration& config) {
        return config.hasOption(keyName);
      }

    protected:

      ///
      /// String identifier of this option.
      std::string keyName;

    };

    ///
    /// Option modifier that can be attached to an Option to constrain its usage.
    ///
    class OptModifier {

    public:

      /// Initialize the modifier with Option it is attached to.
      /// \param opt The Option to attach this modifier to.
      void initialize(Opt* opt) {
        modified = opt;
      }

      /// Verify that the constraint is satisfied.
      /// \param config Configuration.
      /// \return True if the constraint is satisfied.
      virtual bool verify(const Configuration& config) = 0;

      virtual ~OptModifier() = default;

    protected:

      ///
      /// The Option that this modifier was attached to.
      Opt* modified = nullptr;

    };

    ///
    /// Global container for all options. Statically initializes appropriate storage.
    /// All options should be specified as global variables and are then automatically inserted into this container.
    class Options {

    public:

      /// Insert an option into the container. Does not assume ownership of the option.
      /// \param key Identifier of the option.
      /// \param opt Pointer to the option.
      /// \param mods Option modifiers attached to the given option.
      static void addOption(const std::string& key, Opt* opt,
                            std::initializer_list<OptModifier*> mods) {
        options.emplace(key, opt);
        for (auto m : mods) {
          m->initialize(opt);
          activeModifiers.push_back(m);
        }
      }

      ///
      /// Print all available options.
      static void dump() {
        std::cout << "Available options: " << std::endl;
        for (auto& k : options) {
          std::cout << k.first << std::endl;
        }
      }

      /// Parse a Configuration from the option identifiers and values given.
      /// \param input Mapping from option identifier to option value.
      /// \return Configuration.
      static std::shared_ptr<Configuration> parse(const std::map<std::string, std::string>& input) {
        auto config = std::make_shared<Configuration>();
        for (auto& o : input) {
          auto key = o.first;
          auto value = o.second;
          // Try to find the correct option parser for the given identifier.
          // The map avoids linear search for each specified option.
          if (!options.count(key)) {
            SPNC_FATAL_ERROR("Unknown compile option {}", key);
          }
          auto parser = options.at(key);
          // Try to parse the option value using the corresponding parser.
          auto parseResult = parser->parse(key, value);
          if (!parseResult) {
            SPDLOG_WARN("Could not parse option value {} for {}", value, key);
          }
          config->push_back(key, std::move(parseResult.getValue()));
        }
        // Verify all constraints.
        bool verified = true;
        for (auto m : activeModifiers) {
          verified &= m->verify(*config);
        }
        if (!verified) {
          SPNC_FATAL_ERROR("Could not verify configuration constraints!");
        }
        return std::move(config);
      }

      /// Register a new modifier instance in this container, assumes ownership of the modifier.
      /// \tparam Modifier The type of the modifier.
      /// \param mod OptModifier
      /// \return A non-owning pointer to the modifier.
      template<class Modifier>
      static Modifier* registerModifier(std::unique_ptr<Modifier> mod) {
        Modifier* ptr = mod.get();
        allModifiers.push_back(std::move(mod));
        return ptr;
      }

      // DANGER ZONE: Correct static initialization order required.
      ///
      /// Mapping of string identifier to Option (parser).
      static std::unordered_map<std::string, Opt*> options;

      ///
      /// All available, registered modifiers.
      static std::vector<std::unique_ptr<OptModifier>> allModifiers;

      ///
      /// Modifiers attached to registered options.
      static std::vector<OptModifier*> activeModifiers;

      /// Register command-line options to a provided CLI11 app.
      /// \param app Reference to which the options will be registered / added.
      static void registerCLOptions(CLI::App& app);

      /// Collect parsed command-line options and store into a container, which can be used internally.
      /// \param app Reference from which the options will be extracted.
      static std::map<std::string, std::string> collectCLOptions(CLI::App& app);

    };

    /// Configuration interface option.
    /// \tparam Value Type of the value for this option.
    template<typename Value>
    class Option : public Opt {

    public:

      /// Constructor.
      /// \param k String identifier of this option.
      /// \param modifier Constraints to attach to this option.
      explicit Option(std::string k, std::initializer_list<OptModifier*> modifier = {}) noexcept : Opt{std::move(k)} {
        Options::addOption(keyName, this, modifier);
      }

      /// Constructor specifying an default value for this option.
      /// \param k String identifier of this option.
      /// \param defaultVal Default value for this option.
      /// \param modifier Constraints to attach to this option.
      Option(std::string k, Value defaultVal, std::initializer_list<OptModifier*> modifier = {}) noexcept : Opt{
          std::move(k)} {
        hasDefault = true;
        defaultValue = defaultVal;
        Options::addOption(keyName, this, modifier);
      }

      /// Parse the option.
      /// \param key String identifier of the option to parse.
      /// \param value Value to parse.
      /// \return llvm::None if the string identifier did not match, otherwise the parsed value.
      llvm::Optional<std::unique_ptr<OptValue>> parse(const std::string& key,
                                                      const std::string& value) override {
        if (key != keyName) {
          SPDLOG_WARN("Identifier did not match this option!");
          return llvm::None;
        }
        std::unique_ptr<OptValue>
            result = std::make_unique<OptionValue<Value>>(detail::OptionParsers::parse<Value>(value));
        return result;
      }

      /// Retrieve the value specified for this Option from the Configuration.
      /// \param config Configuration.
      /// \return The explicitly specified value. If none was given, the default value. In case this option does not
      /// have a default value, throws an error.
      Value get(const Configuration& config) {
        if (config.hasOption(keyName)) {
          return getVal(config);
        }
        if (hasDefault) {
          return defaultValue;
        }
        SPNC_FATAL_ERROR("Trying to get value of non-present option {}!", keyName)
      }

      /// Retrieve the value specified for this Option from the Configuration or an alternative value.
      /// \param config Configuration.
      /// \param elseValue Value to use if the option was not specified. Supersedes the default-value of the option.
      /// \return The explicitly specified value or the given alternative value.
      Value getOrElse(const Configuration& config, Value elseValue) {
        if (config.hasOption(keyName)) {
          return getVal(config);
        }
        return elseValue;
      }

    protected:

      /// Retrieve value from configuration. Does not check for presence.
      /// \param config Option configuration.
      /// \return Value.
      Value getVal(const Configuration& config) {
        return dynamic_cast<OptionValue<Value>&>(config.get(keyName)).get();
      }

      ///
      /// Indicates if this Option has a default value.
      bool hasDefault = false;

      ///
      /// The default value, if it exists.
      Value defaultValue;

    };

    ///
    /// One possible value for an EnumOpt.
    struct OptionEnumValue {
      ///
      /// String identifer of this enumerated value.
      std::string name;
      ///
      /// Underlying integer value of the enumerated value.
      int value;
      ///
      /// Help description.
      std::string desc;
    };

    // Utility macros to easily define allowed values for an enum option.
#define EnumVal(ENUMVAL, DESC) \
      spnc::interface::OptionEnumValue {#ENUMVAL, int(ENUMVAL), DESC}

#define EnumValN(ENUMVAL, FLAGNAME, DESC) \
      spnc::interface::OptionEnumValue {FLAGNAME, int(ENUMVAL), DESC}

    ///
    /// Special Option that takes a number of predefined, enumerated values as input.
    ///
    class EnumOpt : public Option<int> {

    public:

      /// Constructor.
      /// \param k String identifier for this option.
      /// \param options Allowed values for this option.
      /// \param modifier Constraints to attach to this option.
      EnumOpt(std::string k, std::initializer_list<OptionEnumValue> options,
              std::initializer_list<OptModifier*> modifier = {}) : Option<int>{k, modifier} {
        for (auto& o : options) {
          enumValues.emplace(detail::OptionParsers::toLowerCase(o.name), o);
        }
      }

      /// Constructor additionally specifying a default value.
      /// \tparam D Type of the default value. It must be possible to convert this to int (e.g. enum).
      /// \param k String identifier for this option.
      /// \param defaultVal Default value.
      /// \param options Allowed values for this option.
      /// \param modifier Constraints to attach to this option.
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
          SPDLOG_WARN("Identifier did not match this option!");
          return llvm::None;
        }
        auto v = detail::OptionParsers::toLowerCase(value);
        if (!enumValues.count(v)) {
          // The value does not match any of the enum values.
          SPDLOG_WARN("Specified value {} did not match any of the possible values", v);
          return llvm::None;
        }
        auto id = enumValues.at(v).value;
        std::unique_ptr<OptValue> result = std::make_unique<OptionValue<int>>(id);
        return result;
      }

    private:

      std::unordered_map<std::string, OptionEnumValue> enumValues;

    };

    ///
    /// Modifier to mark an Option as required.
    ///
    class RequiredOpt : public OptModifier {

    public:

      bool verify(const Configuration& config) override {
        return modified->isPresent(config);
      }

    };

    /// Helper-function to create and register a "required"-modifier.
    /// \return Non-owning pointer to the modifier.
    static RequiredOpt* required() {
      return Options::registerModifier(std::make_unique<RequiredOpt>());
    }

    /// Modifier to declare a dependency of an Option on a particular value of another Option.
    /// \tparam OptVal Value type of the Option that this depends on.
    /// \tparam Value Type of the value this option depends on.
    template<typename OptVal, typename Value>
    class Depends : public OptModifier {

    public:

      /// Constructor.
      /// \param opt The Option that this option depends on.
      /// \param depend The value that this option depends on.
      Depends(Option<OptVal>& opt, Value depend) : depOpt{opt}, depVal{depend} {
        static_assert(std::is_constructible<Value, OptVal>::value, "Incompatible type for dependency!");
      }

      bool verify(const Configuration& config) override {
        if (!modified->isPresent(config)) {
          return true;
        }
        if (!depOpt.isPresent(config)) {
          return false;
        }
        OptVal val = depOpt.get(config);
        return (Value{val} == depVal);
      }

    private:

      Option<OptVal>& depOpt;

      Value depVal;

    };

    /// Helper function to create and register a "depends"-constraint.
    /// \tparam OptVal Value type of the Option that this depends on.
    /// \tparam Value Type of the value this option depends on.
    /// \param opt The Option that this option depends on.
    /// \param depend The value that this option depends on.
    /// \return Non-owning pointer to the modifier.
    template<typename OptVal, typename Value>
    static Depends<OptVal, Value>* depends(Option<OptVal>& opt, Value depend) {
      return Options::registerModifier(std::make_unique<Depends < OptVal, Value>>
      (opt, depend));
    }

  }

}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_OPTIONS_H
