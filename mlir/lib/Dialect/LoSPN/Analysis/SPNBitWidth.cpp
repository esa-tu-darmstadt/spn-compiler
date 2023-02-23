#include "LoSPN/Analysis/SPNBitWidth.h"

#include "LoSPN/LoSPNAttributes.h"
#include "llvm/ADT/TypeSwitch.h"


namespace mlir::spn {

void SPNBitWidth::analyzeGraph(Operation* root) {
  using namespace ::mlir::spn::low;

  root->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
      .Case<SPNCategoricalLeaf>([&](SPNCategoricalLeaf op) {
        updateBits(op.getProbabilities().size());
      })
      .Case<SPNHistogramLeaf>([&](SPNHistogramLeaf op) {
        BucketAttr firstBucket = llvm::dyn_cast<BucketAttr>(op.getBuckets()[0]);
        BucketAttr lastBucket = llvm::dyn_cast<BucketAttr>(
          op.getBuckets()[op.getBuckets().size() - 1]
        );

        uint32_t entryCount = lastBucket.getUb() - firstBucket.getLb();
        updateBits(entryCount);
      })
      .Default([](Operation *op) {});
  });

}

void SPNBitWidth::updateBits(uint32_t entryCount) {
  bitsPerVar = std::max(bitsPerVar, llvm::APInt(32, entryCount).ceilLogBase2());
}

SPNBitWidth::SPNBitWidth(Operation* root) {
  analyzeGraph(root);
}

}