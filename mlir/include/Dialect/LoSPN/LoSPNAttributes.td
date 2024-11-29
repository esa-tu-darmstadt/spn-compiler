//==============================================================================
// This file is part of the SPNC project under the Apache License v2.0 by the
// Embedded Systems and Applications Group, TU Darmstadt.
// For the full copyright and license information, please view the LICENSE
// file that was distributed with this source code.
// SPDX-License-Identifier: Apache-2.0
//==============================================================================

#ifndef LoSPN_ATTRIBUTES
#define LoSPN_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "LoSPNDialect.td"

// All of the attributes will extend this class.
class LoSPN_Attr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<LoSPN_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}


// def LoSPN_BSPSuperstepAttr : LoSPN_Attr<"BSPSuperstep", "bsp_superstep"> {
//     let summary = "An attribute that specifies the superstep in the BSP model.";
//     let parameters = (ins "IntegerAttr":$index);

//     let assemblyFormat = "`<` $index `>`";
// }

// def LoSPN_BSPScheduleAttr : LoSPN_Attr<"BSPSchedule", "bsp_schedule"> {
//     let summary = "An attribute that specifies a schedule in the BSP model.";
//     let parameters = (ins ArrayRefParameter<"partitioning::BSPSchedule">:$supersteps);

//     let assemblyFormat = "`<` $supersteps `>`";
// }


def LoSPN_BSPSuperstep : AttrDef<LoSPN_Dialect, "BSPSuperstep"> {
  let summary = "Represents a single superstep in a BSP schedule";
  let description = [{
    A superstep contains a set of tasks that can be executed in parallel, along with
    their processor assignments and timing information.
  }];

  let parameters = (ins 
    "DenseIntElementsAttr":$taskIds,
    "DenseIntElementsAttr":$processorIds
  );

  let mnemonic = "superstep";
  let assemblyFormat = "`<` `{` `task_ids` `=` $taskIds `,` `proc_ids` `=` $processorIds `}` `>`";
}

def LoSPN_BSPSchedule : AttrDef<LoSPN_Dialect, "BSPSchedule"> {
  let summary = "Represents a complete BSP schedule";
  let description = [{
    A BSP (Bulk Synchronous Parallel) schedule organizes parallel computation into
    a sequence of supersteps. Each superstep consists of:
    1) Local computation and communication phase
    2) Synchronization barrier

    The schedule ensures that communication between processors only occurs between
    supersteps and all processors execute supersteps synchronously.
  }];

  let parameters = (ins
    "ArrayAttr":$supersteps // Array of BSP_Superstep
  );

  let mnemonic = "schedule";
  let assemblyFormat = "`<` `{` `supersteps` `=` $supersteps `}` `>`";

  let extraClassDeclaration = [{
  }];
}

#endif // LoSPN_ATTRIBUTES