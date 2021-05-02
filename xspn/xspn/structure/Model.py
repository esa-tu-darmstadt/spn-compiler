# ==============================================================================
#  This file is part of the SPNC project under the Apache License v2.0 by the
#  Embedded Systems and Applications Group, TU Darmstadt.
#  For the full copyright and license information, please view the LICENSE
#  file that was distributed with this source code.
#  SPDX-License-Identifier: Apache-2.0
# ==============================================================================

from spn.structure.Base import Node
import numpy as np


class SPNModel:
    """Thin wrapper around SPN graphs, carrying additional information about the model

    Args: 
        rootNode (Node): Root node of the model's SPN graph
        featureValueType: Type used to encode the input feature values.
        modelName (string, optional): Name of the model.

    Attributes:
        root (Node): Root node of the model's SPN graph.
        featureType: Type used to encode the input feature values.
        name (string): Name of the model, or None.

    """

    def __init__(self, rootNode : Node, featureValueType = np.dtype(np.float64).name, modelName = None):
        self.root = rootNode
        self.featureType = featureValueType
        self.name = modelName
