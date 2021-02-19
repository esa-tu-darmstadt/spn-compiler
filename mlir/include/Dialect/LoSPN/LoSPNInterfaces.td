#ifndef LoSPN_INTERFACES
#define LoSPN_INTERFACES

include "mlir/IR/OpBase.td"

// TODO

def LoSPNVectorizable : OpInterface<"LoSPNVectorizable"> {

    let description = "Interface to potentially vectorizable operations";

    let methods = [
        InterfaceMethod<
            /*desc=*/"Check if the operation can be vectorized",
            /*retTy=*/"bool",
            /*methodName=*/"isVectorizable",
            /*args=*/(ins "unsigned":$vectorFactor),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                return true;
            }]
        >,
        InterfaceMethod<
            /*desc=*/"Enable vectorization of this operation",
            /*retTy=*/"void",
            /*methodName=*/"setVectorized",
            /*args=*/(ins "unsigned":$vectorFactor),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                assert($_op->template hasTrait<OpTrait::spn::low::VectorizableOp>());
                $_op.vectorFactor(vectorFactor);
            }]
        >,
        InterfaceMethod<
            /*desc=*/"Check if the operation should be vectorized",
            /*retTy=*/"bool",
            /*methodName=*/"checkVectorized",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                assert($_op->template hasTrait<OpTrait::spn::low::VectorizableOp>());
                return $_op.isVectorized();
            }]
        >,
    ];

}

#endif // LoSPN_INTERFACES