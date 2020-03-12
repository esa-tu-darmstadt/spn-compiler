//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <driver/Options.h>

using namespace spnc::interface;

// Definitions of the static members of class Options.
std::unordered_map<std::string, Opt*> Options::options;
std::vector<std::unique_ptr<OptModifier>> Options::allModifiers;
std::vector<OptModifier*> Options::activeModifiers;

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