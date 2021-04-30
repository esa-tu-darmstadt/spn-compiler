//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

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
        InterfaceMethod<
            /*desc*/"Get the vector width with which the operation should be vectorized",
            /*retTy=*/"unsigned",
            /*methodName=*/"getVectorWidth",
            /*args=*/(ins),
            /*methodBody=*/[{
                assert($_op->template hasTrait<OpTrait::spn::low::VectorizableOp>());
                return $_op.vectorFactor();
            }],
            /*defaultImplementation=*/""
        >,
    ];

}

def LeafNodeInterface : OpInterface<"LeafNodeInterface"> {

  let description = [{
  Interface common to all LoSPN leaf nodes.
  }];

}

#endif // LoSPN_INTERFACES