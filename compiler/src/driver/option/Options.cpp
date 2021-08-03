//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include <driver/Options.h>

using namespace spnc::interface;

// Definitions of the static members of class Options.
std::unordered_map<std::string, Opt*>& Options::options() {
  static auto* _options = new std::unordered_map<std::string, Opt*>();
  return *_options;
};

std::vector<std::unique_ptr<OptModifier>>& Options::allModifiers() {
  static auto* _modifiers = new std::vector<std::unique_ptr<OptModifier>>();
  return *_modifiers;
};

std::vector<OptModifier*>& Options::activeModifiers() {
  static auto* _modifiers = new std::vector<OptModifier*>();
  return *_modifiers;
};

/// Parse a value from string.
/// \tparam Value Type of the value to parse.
/// \param value String containing the value.
/// \return Parse result.
template<typename Value>
Value detail::OptionParsers::parse(const std::string& value) {
  static_assert(std::is_constructible<Value, std::string>::value, "Must be constructible from string!");
  // As a default, try to construct a the value from a string.
  return Value{value};
}

/// Specialization to parse integer options, using standard library facilities to parse the int from the string.
/// \param value String.
/// \return Integer value.
template<>
int detail::OptionParsers::parse(const std::string& value) {
  return std::stoi(value);
}

/// Specialization to parse unsigned integer options,
/// using standard library facilities to parse the unsigned int from the string.
/// \param value String.
/// \return Unsigned integer value.
template<>
unsigned detail::OptionParsers::parse(const std::string& value) {
  return std::stol(value);
}

/// Specialization to parse floating-point options,
/// using standard library facilities to parse the double from the string.
/// \param value String.
/// \return Double-precision floating-point value.
template<>
double detail::OptionParsers::parse(const std::string& value) {
  return std::stod(value);
}

/// Specialization to parse string options.
/// \param value String.
/// \return The same string.
template<>
std::string detail::OptionParsers::parse(const std::string& value) {
  return value;
}

/// Specialization to parse boolean options. The string is converted to lower case and the returned
/// boolean is true if the string was "true" or "yes" and false otherwise.
/// \param value String.
/// \return True if the string was "true" or "yes", false otherwise.
template<>
bool spnc::interface::detail::OptionParsers::parse(const std::string& value) {
  std::string v = toLowerCase(value);
  return v == "true" || v == "yes";
}

void Options::registerCLOptions(CLI::App& app) {
  // Add positional, 'unnamed' spn file option
  app.add_option("spn")->check(CLI::ExistingFile);
  for (auto& o : options()) {
    std::string opt_name;
    // Adding the same option name prefixed with "--" will result in the option being NON-positional and having a lname.
    opt_name.append(o.first).append(",--").append(o.first);
    app.add_option(opt_name);
  }
}

std::map<std::string, std::string> Options::collectCLOptions(CLI::App& app) {
  std::map<std::string, std::string> opts;

  for (auto& o : app.get_options()) {
    // Only provided options will be returned
    if (!o->get_lnames().empty() && !o->results().empty()) {
      opts.emplace(*o->get_lnames().begin(), *o->results().begin());
    }
  }

  return opts;
}
