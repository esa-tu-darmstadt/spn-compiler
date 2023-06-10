#include "LoSPNtoFPGA2/Conversion.hpp"

#include "LoSPNtoFPGA2/Scheduling.hpp"

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"

#include <firp/lowering.hpp>


namespace mlir::spn::fpga {

using namespace firp;
using namespace low;

struct ConversionSettings {
  ufloat::UFloatConfig ufloatConfig;
  bool use32Bit;
  circt::firrtl::FIRRTLBaseType probType;
  circt::firrtl::FIRRTLBaseType indexType;
};

class Histogram : public Module<Histogram> {
  std::vector<double> probabilities;
  ConversionSettings settings;
public:
  Histogram(SPNHistogramLeaf histogram, uint32_t id, const ConversionSettings& settings):
    Module<Histogram>("Histogram", {
      Port("in", true, settings.indexType),
      Port("out", false, settings.probType)
    }, id), settings(settings) {

    for (Attribute _attr : histogram.getBuckets()) {
      BucketAttr attr = llvm::dyn_cast<BucketAttr>(_attr);
      assert(attr);

      // bounds are [lb, ub)
      int size = attr.getUb() - attr.getLb();
      double prob = attr.getVal().convertToDouble();

      for (int i = 0; i < size; ++i)
        probabilities.push_back(prob);
    }

    build();
  }

  Histogram(SPNCategoricalLeaf categorical, uint32_t id, const ConversionSettings& settings):
    Module<Histogram>("Histogram", {
      Port("in", true, settings.indexType),
      Port("out", false, settings.probType)
    }, id), settings(settings) {

    for (Attribute prob : categorical.getProbabilities())
      probabilities.push_back(prob.dyn_cast<FloatAttr>().getValueAsDouble());

    build();
  }

  void body() {
    std::vector<FValue> probabilityBits;

    for (double prob : probabilities)
      probabilityBits.push_back(uval(
        ufloat::doubleToUFloatBits(prob, settings.ufloatConfig),
        settings.ufloatConfig.getWidth()
      ));

    auto vec = vector(probabilityBits);
    io("out") <<= regNext(vec[io("in")]);
  }
};

class SPNBodyTop : public Module<SPNBodyTop> {
  ConversionSettings settings;
  SPNBody spnBody;
  SchedulingProblem& schedulingProblem;

  static std::vector<Port> getPorts(SPNBody body, const ConversionSettings& settings) {
    std::vector<Port> ports {
      Port("out_prob", false, uintType(settings.use32Bit ? 32 : 64))
    };

    for (std::size_t i = 0; i < body.getOperands().size(); ++i)
      ports.push_back(Port("in_" + std::to_string(i), true, settings.indexType));

    return ports;
  }
public:
  SPNBodyTop(SPNBody body, const ConversionSettings& settings, SchedulingProblem& schedulingProblem):
    Module<SPNBodyTop>("SPNBody", getPorts(body, settings)),
    settings(settings), spnBody(body), schedulingProblem(schedulingProblem) { build(); }

  void body() {
    IRMapping valueMapping;
    uint32_t id = 0; // to make histograms unique

    for (std::size_t i = 0; i < spnBody.getOperands().size(); ++i)
      valueMapping.map(
        spnBody.getRegion().front().getArgument(i),
        io("in_" + std::to_string(i))
      );

    spnBody.walk([&](Operation *op) {
      if (isa<SPNBody>(op))
        return;

      TypeSwitch<Operation *>(op)
        .Case<SPNAdd>([&](SPNAdd op) {
          ufloat::FPAdd add(settings.ufloatConfig);

          // old op result is now associated with add result
          valueMapping.map(op->getResult(0), add.io("c"));

          uint32_t aDelay = schedulingProblem.getDelay(op->getOperand(0), op);
          add.io("a") <<= shiftRegister(valueMapping.lookup(op->getOperand(0)), aDelay);

          uint32_t bDelay = schedulingProblem.getDelay(op->getOperand(1), op);
          add.io("b") <<= shiftRegister(valueMapping.lookup(op->getOperand(1)), bDelay);
        })
        .Case<SPNMul>([&](SPNMul op) { 
          ufloat::FPMult mult(settings.ufloatConfig);

          // old op result is now associated with mult result
          valueMapping.map(op->getResult(0), mult.io("c"));

          uint32_t aDelay = schedulingProblem.getDelay(op->getOperand(0), op);
          mult.io("a") <<= shiftRegister(valueMapping.lookup(op->getOperand(0)), aDelay);

          uint32_t bDelay = schedulingProblem.getDelay(op->getOperand(1), op);
          mult.io("b") <<= shiftRegister(valueMapping.lookup(op->getOperand(1)), bDelay);
        })
        .Case<SPNCategoricalLeaf>([&](SPNCategoricalLeaf op) {
          Histogram hist(op, id++, settings);
          valueMapping.map(op->getResult(0), hist.io("out"));
          uint32_t inDelay = schedulingProblem.getDelay(op->getOperand(0), op);
          hist.io("in") <<= shiftRegister(valueMapping.lookup(op->getOperand(0)), inDelay);
        })
        .Case<SPNHistogramLeaf>([&](SPNHistogramLeaf op) {
          Histogram hist(op, id++, settings);
          valueMapping.map(op->getResult(0), hist.io("out"));
          uint32_t inDelay = schedulingProblem.getDelay(op->getOperand(0), op);
          hist.io("in") <<= shiftRegister(valueMapping.lookup(op->getOperand(0)), inDelay);
        })
        .Case<SPNConstant>([&](SPNConstant op) {
          uint64_t bits = ufloat::doubleToUFloatBits(op.getValue().convertToDouble(), settings.ufloatConfig);
          auto constant = firp::uval(bits, settings.ufloatConfig.getWidth());
          valueMapping.map(op->getResult(0), constant);
        })
        .Case<SPNLog>([&](SPNLog op) {
          // does nothing
          valueMapping.map(op->getResult(0), valueMapping.lookup(op->getOperand(0)));
        })
        .Case<SPNYield>([&](SPNYield op) {
          ufloat::FPConvert convert(settings.ufloatConfig, settings.use32Bit);
          convert.io("in") <<= valueMapping.lookup(op->getOperand(0));
          io("out_prob") <<= convert.io("out");
        })
        .Default([&](Operation *op) {
          llvm_unreachable("unhandled op");
        });
    });   
  }
};

llvm::Optional<mlir::ModuleOp> convert(mlir::ModuleOp modOp, const ConversionOptions& options) {
  uint32_t addDelay = ufloat::scheduling::ufloatFPAddDelay(options.ufloatConfig);
  uint32_t multDelay = ufloat::scheduling::ufloatFPMultDelay(options.ufloatConfig);
  uint32_t convertDelay = ufloat::scheduling::ufloatFPConvertDelay(options.ufloatConfig);

  llvm::outs() << "add delay: " << addDelay << "\n";
  llvm::outs() << "mult delay: " << multDelay << "\n";
  llvm::outs() << "convert delay: " << convertDelay << "\n";
  //assert(false);

  auto getDelayAndType = [&](Operation *op) -> std::tuple<uint32_t, std::string> {
    return TypeSwitch<Operation *, std::tuple<uint32_t, std::string>>(op)
      .Case<SPNAdd>([&](SPNAdd op) { return std::make_tuple(addDelay, "add"); })
      .Case<SPNMul>([&](SPNMul op) { return std::make_tuple(multDelay, "mul"); })
      .Case<SPNCategoricalLeaf>([&](SPNCategoricalLeaf op) { return std::make_tuple(1, "cat"); })
      .Case<SPNHistogramLeaf>([&](SPNHistogramLeaf op) { return std::make_tuple(1, "hist"); })
      .Case<SPNConstant>([&](SPNConstant op) { return std::make_tuple(0, "const"); })
      .Case<SPNLog>([&](SPNLog op) { return std::make_tuple(0, "log"); })
      .Case<SPNYield>([&](SPNYield op) { return std::make_tuple(convertDelay, "convert"); })
      .Default([&](Operation *op) -> std::tuple<uint32_t, std::string> {
        op->dump();
        llvm_unreachable("unhandled op");
      });
      ;
  };

  OpBuilder builder(modOp.getContext());

  ModuleOp newRoot = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&newRoot.getRegion().front());

  firp::attachFirpContext(newRoot, "SPNBody");

  ConversionSettings settings{
    .ufloatConfig = options.ufloatConfig,
    .use32Bit = options.use32Bit,
    .probType = firp::uintType(options.ufloatConfig.getWidth()),
    .indexType = firp::uintType(8)
  };

  modOp.walk([&](SPNBody body) {
    SchedulingProblem schedulingProblem(body);

    schedulingProblem.construct(getDelayAndType);

    if (failed(schedulingProblem.check()))
      assert(false && "Problem check failed!");

    if (failed(circt::scheduling::scheduleASAP(schedulingProblem)))
      assert(false && "Could not schedule problem!");

    circt::scheduling::dumpAsDOT(schedulingProblem, "schedule.dot");

    SPNBodyTop spnBody(body, settings, schedulingProblem);
    FModuleOp topModule = spnBody.makeTop();

    if (failed(firpContext()->finish()))
      assert(false && "firpContext()->finish() failed!");

    uint32_t totalEndTime = schedulingProblem.getTotalEndTime();
    topModule->setAttr("fpga.body_delay", builder.getI32IntegerAttr(totalEndTime));

    if (options.performLowering)
      assert(succeeded(lowerFirrtlToHw()));
  });

  return newRoot;
}

}