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
        CPred<"$_self.isa<::mlir::spn::ProbabilityType>()">, "probability type">,
        BuildableType<"$_builder.getType<::mlir:spn::ProbabilityType>()"> {

    let typeDescription = [{
        `spn.probability_type` abstractly represents a probability type.
    }];

}

#endif // SPN_DIALECT