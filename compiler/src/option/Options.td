include "llvm/Option/OptParser.td"

/////////
// Groups

def Vectorization_Group : OptionGroup<"<vectorization group>">;
def Target_Group : OptionGroup<"<target group>">;

class TargetOpts<string base>
  : KeyPathAndMacro<"TargetOpts->", base, "TARGET_"> {}

def target : Separate<["-"], "target">,
  HelpText<"Specify target triple (e.g. i686-apple-darwin9)">,
  MarshallingInfoString<TargetOpts<"Triple">, "llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple())">,
  AlwaysEmit, Normalizer<"normalizeTriple">;