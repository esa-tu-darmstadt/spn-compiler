#ifndef SPN_DIALECT
#define SPN_DIALECT

include "mlir/IR/OpBase.td"

///
/// Dialect for SPN-specific operations.
///
def SPN_Dialect : Dialect {
    let name = "spn";
    let summary = "An out-of-tree MLIR dialect for Sum-Product Networks";
    let description = [{
        This dialect is specialized to represent Sum-Product Networks and different
        types of queries (e.g. MPE) that can be performed on SPNs.
    }];
    let cppNamespace = "::mlir::spn";
}

///
/// Base operation for all operations of this dialect.
///
class SPN_Op<string mnemonic, list<OpTrait> traits = []> : Op<SPN_Dialect, mnemonic, traits>;

///
/// Dialect-specific type representing probability values.
///
def SPNProbabilityType : DialectType<SPN_Dialect,
        CPred<"$_self.isa<::mlir::spn::ProbabilityType>()">,
        "`spn.probability_type` abstractly represents a probability type.">,
        BuildableType<"$_builder.getType<::mlir:spn::ProbabilityType>()">;

///
/// Type representing any type that can be used to compute probability values.
/// This includes the abstract SPNProbabilityType, any float and integer type and vectors of float/integer.
///
def AnyProbabilityType : DialectType<SPN_Dialect,
                        Or<[SPNProbabilityType.predicate, AnyFloat.predicate,
                          VectorOf<[AnyFloat]>.predicate, AnyInteger.predicate, VectorOf<[AnyInteger]>.predicate]>,
                        "Any type that can be used to represent probabilities of varying precision">;

///
/// Type representing possible input types to the SPN.
/// Currently allows any float or signless integer type.
/// Type conversion are implicit in the leaf nodes, if necessary.
///
def SPNInputType : AnyTypeOf<[AnyFloat, AnySignlessInteger]>;

#endif // SPN_DIALECT