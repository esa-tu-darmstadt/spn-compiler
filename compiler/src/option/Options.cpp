//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include <memory>
#include <option/Options.h>

using namespace spnc::interface;

// Definitions of the static members of class Options.
std::unordered_map<std::string, Opt *> &Options::options() {
  static auto *_options = new std::unordered_map<std::string, Opt *>();
  return *_options;
};

std::vector<std::unique_ptr<OptModifier>> &Options::allModifiers() {
  static auto *_modifiers = new std::vector<std::unique_ptr<OptModifier>>();
  return *_modifiers;
};

std::vector<OptModifier *> &Options::activeModifiers() {
  static auto *_modifiers = new std::vector<OptModifier *>();
  return *_modifiers;
};

/// Parse a value from string.
/// \tparam Value Type of the value to parse.
/// \param value String containing the value.
/// \return Parse result.
template <typename Value>
Value detail::OptionParsers::parse(const std::string &value) {
  static_assert(std::is_constructible<Value, std::string>::value,
                "Must be constructible from string!");
  // As a default, try to construct a the value from a string.
  return Value{value};
}

/// Specialization to parse integer options, using standard library facilities
/// to parse the int from the string. \param value String. \return Integer
/// value.
template <> int detail::OptionParsers::parse(const std::string &value) {
  return std::stoi(value);
}

/// Specialization to parse unsigned integer options,
/// using standard library facilities to parse the unsigned int from the string.
/// \param value String.
/// \return Unsigned integer value.
template <> unsigned detail::OptionParsers::parse(const std::string &value) {
  return std::stol(value);
}

/// Specialization to parse floating-point options,
/// using standard library facilities to parse the double from the string.
/// \param value String.
/// \return Double-precision floating-point value.
template <> double detail::OptionParsers::parse(const std::string &value) {
  return std::stod(value);
}

/// Specialization to parse string options.
/// \param value String.
/// \return The same string.
template <> std::string detail::OptionParsers::parse(const std::string &value) {
  return value;
}

/// Specialization to parse boolean options. The string is converted to lower
/// case and the returned boolean is true if the string was "true" or "yes" and
/// false otherwise. \param value String. \return True if the string was "true"
/// or "yes", false otherwise.
template <>
bool spnc::interface::detail::OptionParsers::parse(const std::string &value) {
  std::string v = toLowerCase(value);
  return v == "true" || v == "yes";
}

std::vector<llvm::cl::Option *> Options::registerCLOptions() {
  // global storage for options and categories
  static std::vector<std::unique_ptr<llvm::cl::OptionCategory>>
      clCategoryStorage;
  static std::vector<std::unique_ptr<llvm::cl::Option>> clOptionStorage;

  // lambda to get or add a CL category with a given name
  auto getOrAddCategory =
      [](llvm::StringRef name) -> llvm::cl::OptionCategory & {
    auto it = std::find_if(
        clCategoryStorage.begin(), clCategoryStorage.end(),
        [&name](const std::unique_ptr<llvm::cl::OptionCategory> &cat) {
          return cat->getName() == name;
        });
    if (it != clCategoryStorage.end()) {
      return **it;
    } else {
      auto cat = std::make_unique<llvm::cl::OptionCategory>(name);
      clCategoryStorage.push_back(std::move(cat));
      return *clCategoryStorage.back();
    }
  };

  std::vector<llvm::cl::Option *> clOptions;

  // register the positional argument
  clOptionStorage.push_back(std::make_unique<llvm::cl::opt<std::string>>(
      llvm::cl::Positional, llvm::cl::desc("<input spn>"), llvm::cl::Required,
      llvm::cl::cat(getOrAddCategory("Compilation"))));
  clOptions.push_back(clOptionStorage.back().get());

  // register the optional options
  llvm::StringMap<llvm::cl::Option *> existingOptions =
      llvm::cl::getRegisteredOptions();

  for (auto &pair : options()) {
    auto *option = pair.second;
    // check if the option is already registered
    if (existingOptions.count(option->getKey())) {
      SPDLOG_INFO("Reusing existing option {}", option->getKey());
      clOptions.push_back(existingOptions[option->getKey()]);
      continue;
    }
    auto clOption = std::make_unique<llvm::cl::opt<std::string>>(
        llvm::StringRef(option->getKey()),
        llvm::cl::desc(option->getDescription()),
        llvm::cl::cat(getOrAddCategory(option->getCategory())));
    clOptionStorage.push_back(std::move(clOption));
    clOptions.push_back(clOptionStorage.back().get());
  }

  // hide unrelated options
  std::vector<const llvm::cl::OptionCategory *> relatedCategoryPtrs;
  for (auto &cat : clCategoryStorage)
    relatedCategoryPtrs.push_back(cat.get());
  // llvm::cl::HideUnrelatedOptions(relatedCategoryPtrs);
  return clOptions;
}

std::map<std::string, std::string>
Options::collectCLOptions(const std::vector<llvm::cl::Option *> &clOptions) {
  std::map<std::string, std::string> opts;

  for (auto &clOption : clOptions) {
    auto *clOpt = dynamic_cast<llvm::cl::opt<std::string> *>(clOption);
    if(!clOpt) {
      SPDLOG_WARN("Skipping option {}", clOption->ArgStr);
      continue;
    }
    if (clOpt->getNumOccurrences() == 0)
      continue;
    clOpt->isPositional()
        ? opts["input"] = clOpt->getValue()
        : opts[std::string(clOpt->ArgStr)] = clOpt->getValue();
  }

  return opts;
}

void Options::setCLOptions(const std::map<std::string, std::string> &options) {
  auto &clOptions = llvm::cl::getRegisteredOptions();
  for (auto &[key, value] : options) {
    // check if the option is registered as a command line option
    if (!clOptions.count(key))
      continue;

    auto *clOpt = clOptions[key];
    clOpt->addOccurrence(0, key, value);
  }
}