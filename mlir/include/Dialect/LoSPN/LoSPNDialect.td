//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef LoSPN_DIALECT
#define LoSPN_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

///
/// Dialect for low-level abstractions representing computations used to
/// realize queries on Sum-Product Networks.
///
def LoSPN_Dialect : Dialect {
    let name = "lo_spn";
    let summary = "Out-of-tree MLIR dialect for low-level representation of computations on SPNs";
    let description = [{
        This dialect represents computations used to realize queries on Sum-Product Networks at a
        low level of abstraction.
    }];
    let cppNamespace = "::mlir::spn::low";
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        /// Register all SSP attributes.
        void registerAttributes();
    }];
    let hasConstantMaterializer = 1;
}

///
/// Base operation for all operations of this dialect.
///
class LoSPN_Op<string mnemonic, list<Trait> traits = []> : Op<LoSPN_Dialect, mnemonic, traits>;

///
/// Type representing possible input types to the SPN.
/// Currently allows any float or signless integer type.
/// Type conversion are implicit in the leaf nodes, if necessary.
///
def LoSPNInputType : AnyTypeOf<[AnyFloat, AnySignlessInteger]>;

///
/// Type wrapping an base-type to indicate computation in log-space
///
def LoSPN_IsLogType : CPred<"$_self.isa<::mlir::spn::low::LogType>()">;
def LoSPN_LogType : TypeDef<LoSPN_Dialect, "Log"> {
    // TODO: We currently only support float types in log-space
    // Constraints should be added to the type system to ensure this.

    let parameters = (ins "Type":$baseType);
    let mnemonic = "log";
    let assemblyFormat = "`<` $baseType `>`";
}

///
/// Type representing types that can be used internally for computation
/// of a query. Currently only allows float or integer types.
///
/// TODO: Extend for vector types.
def LoSPNComputeType : AnyTypeOf<[AnyFloat, AnySignlessInteger, LoSPN_LogType]>;


#endif // LoSPN_DIALECT