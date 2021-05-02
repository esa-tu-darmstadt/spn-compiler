#ifndef SPNOpTraits
#define SPNOpTraits

include "SPNDialect.td"

///
/// Trait which can be attached to operations to mark them as vectorizable (such as sums or products).
///
def Vectorizable : NativeOpTrait<"spn::Vectorizable">;

#endif // SPNOpTraits