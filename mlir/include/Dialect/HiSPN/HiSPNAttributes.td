//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef HiSPN_ATTRIBUTES
#define HiSPN_ATTRIBUTES

include "HiSPNDialect.td"
include "mlir/IR/AttrTypeBase.td"

def HistBucketAttr : AttrDef<HiSPN_Dialect, "HistBucket"> {
  let parameters = (ins
    "int32_t":$lowerBound,
    "int32_t":$upperBound,
    "APFloat":$histValue
  );
  let mnemonic = "bucket";
  let assemblyFormat = "`<` $lowerBound `to` $upperBound `=` $histValue `>`";
}

def HistBucketList : TypedArrayAttrBase<HistBucketAttr, "histogram bucket list">;
#endif // HiSPN_ATTRIBUTES