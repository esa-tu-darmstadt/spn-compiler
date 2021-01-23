#ifndef SPNOpTraits
#define SPNOpTraits

include "SPNDialect.td"

///
/// Trait which can be attached to operations to mark them as vectorizable (such as sums or products).
///
def Vectorizable : NativeOpTrait<"spn::Vectorizable">;

///
/// Trait which can be attached to operations to mark them as binarizable (such as weighted sums or products).
///
def Binarizable : NativeOpTrait<"spn::Binarizable">;

#endif // SPNOpTraits