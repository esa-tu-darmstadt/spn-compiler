#include "EmbedController.hpp"

#include "circt/Dialect/FIRRTL/FIRParser.h"

#include "spdlog/fmt/fmt.h"
#include "util/Command.h"
#include <filesystem>


namespace spnc {

std::optional<ModuleOp> EmbedController::generateController() {
  // call scala CLI app
  std::string cmdArgs =
    fmt::format("-in-width {} -out-width {} -body-depth {} -pre-fifo-depth {} -post-fifo-depth {}",
      config.inputBitWidth,
      config.outputBitWidth,
      bodyDelay,
      preFifoDepth,
      postFifoDepth
    );

  std::string cmdString = config.generatorPath.string() + cmdArgs;
  std::string result = spnc::Command::executeExternalCommandAndGetOutput(cmdString);

  // importFIRFile(???, ???, ???);

  return std::nullopt;
}

std::optional<HWModuleOp> EmbedController::getUniqueBody(ModuleOp root) {
  uint32_t count = 0;
  HWModuleOp body;

  root.walk([&](HWModuleOp op) {
    // check name
    if (op.getName() == "spn_body") {
      body = op;
      ++count;
    }
  });

  if (count != 1)
    return std::nullopt;

  return body;
}

bool EmbedController::insertBodyIntoController(ModuleOp controller, HWModuleOp body) {
  // find the external module
  // instantiate hw instance with a reference to our body
  return false;
}

ExecutionResult EmbedController::executeStep(ModuleOp *root) {
  std::optional<HWModuleOp> spnBody = getUniqueBody(*root);

  if (!spnBody.has_value())
    return failure(
      "EmbedController: spn_body must be unique."
    );

  Operation *op = spnBody.value().getOperation();

  if (!op->hasAttr("fpga.body_delay"))
    return failure(
      "EmbedController: spn_body does not have an attribute fpga.body_delay."
    );

  IntegerAttr attr = llvm::dyn_cast<IntegerAttr>(
    op->getAttr("fpga.body_delay")
  );

  if (!attr)
    return failure(
      "EmbedController: fpga.body_delay is not an integer."
    );

  bodyDelay = attr.getInt();
  preFifoDepth = bodyDelay * 2;
  postFifoDepth = bodyDelay * 2;

  std::optional<ModuleOp> firController = generateController();

  if (!firController.has_value())
    return failure(
      "EmbedController: generateController() has failed."
    );

  if (!insertBodyIntoController(firController.value(), spnBody.value()))
    return failure(
      "EmbedController: Could not insert body into controller."
    );

  return failure("EmbedController is not implemented!");
}

}