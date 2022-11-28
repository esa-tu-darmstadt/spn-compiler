//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef HiSPN_DIALECT
#define HiSPN_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

///
/// Dialect for high-level abstractions representing SPNs and queries.
///
def HiSPN_Dialect : Dialect {
    let name = "hi_spn";
    let summary = "Out-of-tree MLIR dialect for high-level representation of SPNs";
    let description = [{
        This dialect represents Sum-Product Networks and queries operating on SPNs at a
        high level of abstraction.
    }];
    let cppNamespace = "::mlir::spn::high";
    //let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        /// Register all SSP attributes.
        void registerAttributes();
    }];
}

///
/// Base operation for all operations of this dialect.
///
class HiSPN_Op<string mnemonic, list<Trait> traits = []> : Op<HiSPN_Dialect, mnemonic, traits>;

///
/// Dialect-specific type representing probability values.
///
def HiSPNProbabilityType : DialectType<HiSPN_Dialect,
        CPred<"$_self.isa<::mlir::spn::high::ProbabilityType>()">,
        "`hi_spn.probability_type` abstractly represents a probability type.">,
        BuildableType<"$_builder.getType<::mlir::spn::high::ProbabilityType>()">;

///
/// Type representing possible input types to the SPN.
/// Currently allows any float or signless integer type.
/// Type conversion are implicit in the leaf nodes, if necessary.
///
def HiSPNInputType : AnyTypeOf<[AnyFloat, AnySignlessInteger]>;

#endif // HiSPN_DIALECT