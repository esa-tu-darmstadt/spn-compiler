# Notes

- LLVM commit: cf15ccdeb6d5254ee7d46c7535c29200003a3880

# Changes

- set c++ standard to 17
- rename `MLIRMath` to `MLIRMathDialect`
- `StructAttr` was replaced by `AttrDef`
  - the options `-gen-struct-attr-decls` and `-gen-struct-attr-defs` no longer exist
  - replaced with `--gen-attrdef-decls` and `--gen-attrdef-defs` (I guess)
  - see:
    - https://reviews.llvm.org/D127375
    - https://discourse.llvm.org/t/psa-structattr-is-deprecated-and-is-being-removed/63068
    - `mlir-tblgen --help`
- `OpTrait` was replaced by `Trait`
  - compare mlir/include/mlir/IR/OpBase.td
- `I32EnumAttrCase` moved from `mlir/include/mlir/IR/OpBase.td` to `mlir/include/mlir/IR/EnumAttr.td`
- replace `let verifier = ...` with `let hasVerifier = 1;`
  - see https://reviews.llvm.org/D118822
- `NoSideEffect` was renamed to `NoMemoryEffect`
- `SameOperandsAndResultType` moved from `mlir/include/mlir/IR/OpBase.td` to `mlir/include/mlir/Interfaces/InferTypeOpInterface.td`
- `7ceffae18c43d0752741ed9ea80d2d7ee4daa70b` converted `OpTrait::FunctionLike` to `FunctionOpInterface`
- `RecursiveSideEffects` to `RecursiveMemoryEffect`
- set to `kEmitAccessorPrefix_Raw` to avoid name clashes between attribute names and interface parameters
- implement histogram buckets as AttrDef

# Convert StructAttr to AttrDef

This can be found in `mlir/include/mlir/Dialect/Tosa/IR/TosaOpBase.td`.

Old:
```
def Tosa_UnaryOpQuantizationAttr : StructAttr<"UnaryOpQuantizationAttr",
  Tosa_Dialect, [
    StructFieldAttr<"input_zp",         I32Attr>,
    StructFieldAttr<"output_zp",        I32Attr>
    ]> {
  let summary = "Attribute for UnaryOp quantization information.";
}
```

New:
```
class Tosa_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Tosa_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def Tosa_UnaryOpQuantizationAttr
    : Tosa_Attr<"UnaryOpQuantization", "unary_quant"> {
  let summary = "Attribute for UnaryOp quantization information.";
  let parameters = (ins "int64_t":$input_zp, "int64_t":$output_zp);
  let assemblyFormat = "`<` struct(params) `>`";
}
```