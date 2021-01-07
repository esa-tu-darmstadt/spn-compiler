//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_DRIVER_ACTION_EMITOBJECTCODE_H
#define SPNC_COMPILER_SRC_DRIVER_ACTION_EMITOBJECTCODE_H

#include <util/FileSystem.h>
#include <driver/Actions.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

namespace spnc {

  ///
  /// Action to translate LLVM IR module into object code for the native CPU target.
  /// The object code is written to the specified object file (*.o)
  class EmitObjectCode : public ActionSingleInput<llvm::Module, ObjectFile> {

  public:

    /// Constructor.
    /// \param _module Action providing the input LLVM IR module.
    /// \param outputFile File to write the object output to.
    /// \param targetMachine LLVM TargetMachine.
    EmitObjectCode(ActionWithOutput<llvm::Module>& _module, ObjectFile outputFile,
                   std::shared_ptr<llvm::TargetMachine> targetMachine);

    ObjectFile& execute() override;

  private:

    ObjectFile outFile;

    std::shared_ptr<llvm::TargetMachine> machine;

    bool cached = false;

  };

}

#endif //SPNC_COMPILER_SRC_DRIVER_ACTION_EMITOBJECTCODE_H
