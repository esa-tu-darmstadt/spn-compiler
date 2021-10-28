//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_PIPELINECONTEXT_H
#define SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_PIPELINECONTEXT_H

#include <memory>
#include "llvm/ADT/DenseMap.h"

namespace spnc {

  struct ElementBase {

    virtual ~ElementBase() = default;

  };

  template<class Content>
  class Element : public ElementBase {

  public:

    explicit Element(std::unique_ptr<Content>&& storeContent) : content{std::move(storeContent)} {};

    Content* get() {
      return content.get();
    }

  private:

    std::unique_ptr<Content> content;

  };

  class PipelineContext {

  public:

    template<typename Content>
    void add(std::unique_ptr<Content>&& element) {
      // TODO Check if already present.
      auto id = typeID<Content>();
      auto container = std::make_unique<Element<Content>>(std::move(element));
      elements.insert({id, std::move(container)});
    }

    template<typename Content, typename ... Args>
    void add(Args&& ... args) {
      auto content = std::make_unique<Content>(std::forward<Args>(args)...);
      add(std::move(content));
    }

    template<typename Content>
    void invalidate() {
      auto id = typeID<Content>();
      if (elements.count(id)) {
        elements.erase(id);
      }
    }

    template<typename Content>
    Content* get() {
      // TODO Check for presence.
      auto id = typeID<Content>();
      auto& element = elements[id];
      auto content = static_cast<Element<Content>*>(element.get());
      return content->get();
    }

    template<typename Content>
    void override(std::unique_ptr<Content>&& content) {
      // TODO Check presence?
      invalidate<Content>();
      add<Content>(std::move(content));
    }

    template<typename Content, typename ... Args>
    void override(Args&& ... args) {
      // TODO Check presense?
      invalidate<Content>();
      add<Content, Args...>(std::forward<Args>(args)...);
    }

    template<typename Content>
    bool has() {
      auto id = typeID<Content>();
      return elements.count(id);
    }

  private:

    llvm::DenseMap<unsigned, std::unique_ptr<ElementBase>> elements;

    template<typename Content>
    static inline unsigned typeID() {
      static unsigned ID = lastTypeID++;
      return ID;
    }

    static unsigned lastTypeID;

  };

}

#endif //SPNC_COMPILER_INCLUDE_DRIVER_PIPELINE_PIPELINECONTEXT_H
