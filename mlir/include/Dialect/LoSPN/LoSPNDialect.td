#ifndef LoSPN_DIALECT
#define LoSPN_DIALECT

include "mlir/IR/OpBase.td"

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
}

///
/// Base operation for all operations of this dialect.
///
class LoSPN_Op<string mnemonic, list<OpTrait> traits = []> : Op<LoSPN_Dialect, mnemonic, traits>;

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
def LoSPN_LogType : DialectType<LoSPN_Dialect, LoSPN_IsLogType, "log", "::mlir::spn::low::LogType">;

///
/// Type representing types that can be used internally for computation
/// of a query. Currently only allows float or integer types.
///
/// TODO: Extend for vector types.
def LoSPNComputeType : AnyTypeOf<[AnyFloat, AnySignlessInteger, LoSPN_LogType]>;


#endif // LoSPN_DIALECT