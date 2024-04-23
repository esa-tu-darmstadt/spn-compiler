//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_PIPELINECONTEXT_H
#define SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_PIPELINECONTEXT_H

#include "util/Logging.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace spnc {

///
/// Base class for elements that can be inserted into a PipelineContext.
struct ElementBase {

  virtual ~ElementBase() = default;
};

/// Wrapper for elements to be inserted into PipelineContext.
/// \tparam Content Type of the actual content of the element.
template <class Content>
class Element : public ElementBase {

public:
  /// Constructor
  /// \param storeContent Owning pointer to the content of the element.
  explicit Element(std::unique_ptr<Content> &&storeContent)
      : content{std::move(storeContent)} {};

  /// Get the content.
  /// \return Non-owning pointer to the content.
  Content *get() { return content.get(); }

private:
  std::unique_ptr<Content> content;
};

///
/// Context to be attached to a Pipeline to retain additional information across
/// different steps of the compilation pipeline.
class PipelineContext {

public:
  /// Add an element to the context.
  /// \tparam Content Type of the content.
  /// \param element Owning pointer to the content.
  template <typename Content>
  void add(std::unique_ptr<Content> &&element) {
    if (has<Content>()) {
      // Check whether already present.
      SPNC_FATAL_ERROR("Cannot add, element of type {} already present. Use "
                       "override instead",
                       typeid(Content).name())
    }
    auto id = typeID<Content>();
    auto container = std::make_unique<Element<Content>>(std::move(element));
    elements.insert({id, std::move(container)});
  }

  /// Construct and add an element to the context.
  /// \tparam Content Type of the content.
  /// \tparam Args Type of the content's constructor arguments.
  /// \param args Parameters to the content's constructor.
  template <typename Content, typename... Args>
  void add(Args &&...args) {
    auto content = std::make_unique<Content>(std::forward<Args>(args)...);
    add(std::move(content));
  }

  /// Invalidate an existing element.
  /// \tparam Content Type of the element to invalidate.
  template <typename Content>
  void invalidate() {
    auto id = typeID<Content>();
    if (elements.count(id)) {
      elements.erase(id);
    }
  }

  /// Get the element of the specified type.
  /// \tparam Content Type of the content.
  /// \return Non-owning pointer to the content.
  template <typename Content>
  Content *get() {
    if (!has<Content>()) {
      // Check for presence.
      SPNC_FATAL_ERROR("Cannot get, no element of type {} present in context",
                       typeid(Content).name());
    }
    auto id = typeID<Content>();
    auto &element = elements[id];
    auto content = static_cast<Element<Content> *>(element.get());
    return content->get();
  }

  /// Override an existing element with new content in the context.
  /// \tparam Content Type of the content.
  /// \param content Owning pointer to the content.
  template <typename Content>
  void override(std::unique_ptr<Content> &&content) {
    if (!has<Content>()) {
      SPDLOG_WARN(
          "No element of type {} present, performing add instead of override",
          typeid(Content).name());
    }
    invalidate<Content>();
    add<Content>(std::move(content));
  }

  /// Construct a new content and override an existing element in the context.
  /// \tparam Content Type of the content.
  /// \tparam Args Type of the content's constructor arguments.
  /// \param args Parameters to the content's constructor.
  template <typename Content, typename... Args>
  void override(Args &&...args) {
    if (!has<Content>()) {
      SPDLOG_WARN(
          "No element of type {} present, performing add instead of override",
          typeid(Content).name());
    }
    invalidate<Content>();
    add<Content, Args...>(std::forward<Args>(args)...);
  }

  /// Check if an element of the given type is present in the context.
  /// \tparam Content Type of the content.
  /// \return true if an element of the given type is present, false otherwise.
  template <typename Content>
  bool has() {
    auto id = typeID<Content>();
    return elements.count(id);
  }

private:
  llvm::DenseMap<unsigned, std::unique_ptr<ElementBase>> elements;

  template <typename Content>
  static inline unsigned typeID() {
    static unsigned ID = lastTypeID++;
    return ID;
  }

  // Single counter to yield unique IDs per type.
  static unsigned lastTypeID;
};

} // namespace spnc

#endif // SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_PIPELINECONTEXT_H
