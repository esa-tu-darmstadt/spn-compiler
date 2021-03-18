import logging
import os
import numpy as np
import capnp

from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, rebuild_scopes_bottom_up, assign_ids, get_number_of_nodes
from spn.structure.StatisticalTypes import Type, MetaType
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.parametric.Parametric import Gaussian, Categorical

from xspn.structure.Model import SPNModel
from xspn.structure.Query import Query, JointProbability, ErrorModel, ErrorKind

# Magic import making the schema defined in the schema language available
from  xspn.serialization.binary.capnproto import spflow_capnp

logger = logging.getLogger(__name__)

metaType2Enum = {MetaType.REAL : "real", MetaType.BINARY : "binary", MetaType.DISCRETE : "discrete"}

enum2MetaType = {v : k for k, v in metaType2Enum.items()}

type2Enum = {Type.REAL : "real", 
                Type.INTERVAL : "interval", 
                Type.POSITIVE : "positive", 
                Type.CATEGORICAL : "categorical",
                Type.ORDINAL : "ordinal", 
                Type.COUNT : "count", 
                Type.BINARY : "binary"}

enum2Type = {v : k for k, v in type2Enum.items()}

class ListHandler:

    def __init__(self, list):
        self._list = list
        self._index = 0

    def getElement(self):
        element = self._list[self._index]
        print(f"Filling list element {self._index}")
        self._index = self._index + 1
        return element

class BinarySerializer:
    """Interface to serialize SPNs from SPFlow into an efficient binary format."""

    def __init__(self, fileName, bufferSize = 10 * (2**10), clearFile = True):
        """Initialize the serializer.

        Keyword arguments:
        fileName -- The name of the output file.
        bufferSize -- Buffer size used during writing.
        clearFile -- If set to True (default), completely erase the file before writing.
        """
        self.assignedIDs = {}
        self.fileName = fileName
        self.bufferSize = bufferSize
        if clearFile:
            # Clear the content of the file if not instructed otherwise.
            open(fileName, "w").close()

    def serialize_to_file(self, content):
        with open(self.fileName, "a+b", buffering=self.bufferSize*(2**10)) as outFile:
            header = spflow_capnp.Header.new_message()
            if isinstance(content, SPNModel):
                header.model = self._serialize_model(content)
            elif isinstance(content, Query):
                header.query = self._serialize_query(content)
                serializer = self._serialize_query
            else:
                raise NotImplementedError(f"No serialization defined for content {content} of type {type(content)}")
            header.write(outFile)

    def _serialize_query(self, query):
        query_msg = spflow_capnp.Query.new_message()
        query_msg.batchSize = query.batchSize
        if query.errorModel.kind is ErrorKind.ABSOLUTE:
            query_msg.errorKind = "absolute"
        elif query.errorModel.kind is ErrorKind.RELATIVE:
            query_msg.errorKind = "relative"
        else:
            raise NotImplementedError(f"No serialization defined for error kind {query.errorModel.kind}")
        query_msg.maxError = query.errorModel.error
        if isinstance(query, JointProbability):
            query_msg.joint = self._serialize_joint(query)
        else:
            raise NotImplementedError(f"No serialization defined for query {query} of type {type(query)}")
        return query_msg

    def _serialize_joint(self, joint):
        joint_msg = spflow_capnp.JointProbability.new_message()
        joint_msg.model = self._serialize_model(joint.graph)
        joint_msg.supportMarginal = joint.supportsMarginal()
        return joint_msg


    def _serialize_model(self, model):
        msg = spflow_capnp.Model.new_message()
        assert is_valid(model.root), "SPN invalid before serialization"
        # Assign (new) IDs to the nodes
        # Keep track of already assigned IDs, so the IDs are 
        # unique for the whole file.
        assign_ids(model.root, self.assignedIDs)
        # Rebuild scopes bottom-up
        rebuild_scopes_bottom_up(model.root)
        msg.rootNode = model.root.id
        msg.numFeatures = len(model.root.scope)
        msg.featureType = model.featureType
        scope = msg.init("scope", len(model.root.scope))
        for i,v in enumerate(model.root.scope):
            scope[i] = self._unwrap_value(v)
        name = ""
        if model.name is not None:
            name = model.name
        msg.name = name
        numNodes = get_number_of_nodes(model.root)
        nodes = msg.init("nodes", numNodes)
        nodeList = ListHandler(nodes)
        self._serialize_graph([model.root], nodeList)
        return msg


    def _serialize_graph(self, rootNodes, nodeList):
        """Serialize SPN graphs to binary format. SPN graphs are given by their root node."""
        # Buffering write, buffer size was specified at initialization (defaults to 10 MiB).
        # The buffer size is specified in KiB during initialization, scale to bytes here.
        numNodes = 0
        for spn in rootNodes:
            visited = set()
            self._binary_serialize(spn, True, visited, nodeList)
            numNodes += len(visited)
        print(f"Serialized {numNodes} nodes to {self.fileName}")

    def _binary_serialize(self, node, is_rootNode, visited_nodes, nodeList):
        if node.id not in visited_nodes:
            if isinstance(node, Product):
                nodeList = self._serialize_product(node, is_rootNode, visited_nodes, nodeList)
            elif isinstance(node, Sum):
                nodeList = self._serialize_sum(node, is_rootNode, visited_nodes, nodeList)
            elif isinstance(node, Histogram):
                nodeList = self._serialize_histogram(node, is_rootNode, visited_nodes, nodeList)
            elif isinstance(node,Gaussian):
                nodeList = self._serialize_gaussian(node, is_rootNode, visited_nodes, nodeList)
            elif isinstance(node, Categorical):
                nodeList = self._serialize_categorical(node, is_rootNode, visited_nodes, nodeList)
            else:
                raise NotImplementedError(f"No serialization defined for node {node} of type {type(node)}")
            visited_nodes.add(node.id)
            return nodeList

    def _serialize_product(self, product, is_rootNode, visited_nodes, nodeList):
        # Serialize child nodes before node itself
        for c in product.children:
            self._binary_serialize(c, False, visited_nodes, nodeList)
        # Construct inner product node message.
        prod_msg = spflow_capnp.ProductNode.new_message()
        children = prod_msg.init("children", len(product.children))
        for i, child in enumerate(product.children):
            children[i] = child.id
        # Construct surrounding node message
        node = nodeList.getElement()
        node.id = product.id
        node.product = prod_msg
        node.rootNode = is_rootNode

    def _serialize_sum(self, sum, is_rootNode, visited_nodes, nodeList):
        # Serialize child nodes before node itself
        for c in sum.children:
            self._binary_serialize(c, False, visited_nodes, nodeList)
        # Construct innner sum node message
        sum_msg = spflow_capnp.SumNode.new_message()
        children = sum_msg.init("children", len(sum.children))
        for i, child in enumerate(sum.children):
            children[i] = child.id
        weights = sum_msg.init("weights", len(sum.weights))
        for i, w in enumerate(sum.weights):
            weights[i] = BinarySerializer._unwrap_value(w)
        # Construct surrounding node message
        node = nodeList.getElement()
        node.id = sum.id
        node.sum = sum_msg
        node.rootNode = is_rootNode

    def _serialize_histogram(self, hist, is_rootNode, visited_nodes, nodeList):
        # Construct inner histogram leaf message.
        hist_msg = spflow_capnp.HistogramLeaf.new_message()
        breaks = hist_msg.init("breaks", len(hist.breaks))
        for i,b in enumerate(hist.breaks):
            breaks[i] = int(b)
        densities = hist_msg.init("densities", len(hist.densities))
        for i,d in enumerate(hist.densities):
            densities[i] = BinarySerializer._unwrap_value(d)
        reprPoints = hist_msg.init("binReprPoints", len(hist.bin_repr_points))
        for i,r in enumerate(hist.bin_repr_points):
            reprPoints[i] = BinarySerializer._unwrap_value(r)
        hist_msg.type = type2Enum.get(hist.type)
        hist_msg.metaType = metaType2Enum.get(hist.meta_type)
        # Check that scope is defined over a single variable
        assert len(hist.scope) == 1, "Expecting Histogram to be univariate"
        hist_msg.scope = BinarySerializer._unwrap_value(hist.scope[0])
        # Construct surrounding node message.
        node = nodeList.getElement()
        node.hist = hist_msg
        node.rootNode = is_rootNode
        node.id = hist.id

    def _serialize_gaussian(self, gauss, is_rootNode, visited_nodes, nodeList):
        # Construct inner Gaussian leaf message
        gauss_msg = spflow_capnp.GaussianLeaf.new_message()
        gauss_msg.mean = BinarySerializer._unwrap_value(gauss.mean)
        gauss_msg.stddev = BinarySerializer._unwrap_value(gauss.stdev)
        # Check that scope is defined over a single variable
        assert len(gauss.scope) == 1, "Expecting Gauss to be univariate"
        gauss_msg.scope = BinarySerializer._unwrap_value(gauss.scope[0])
        # Construct surrounding node message.
        node = nodeList.getElement()
        node.gaussian = gauss_msg
        node.rootNode = is_rootNode
        node.id = gauss.id

    def _serialize_categorical(self, categorical, is_rootNode, visited_nodes, nodeList):
        # Construct inner categorical leaf message.
        cat_msg = spflow_capnp.CategoricalLeaf.new_message()
        probabilities = cat_msg.init("probabilities", len(categorical.p))
        for i,p in enumerate(categorical.p):
            probabilities[i] = BinarySerializer._unwrap_value(p)
        # Check that the scope is defined over a single variable
        assert len(categorical.scope) == 1, "Expecting Categorical leaf to be univariate"
        cat_msg.scope = BinarySerializer._unwrap_value(categorical.scope[0])
        node = nodeList.getElement()
        node.categorical = cat_msg
        node.rootNode = is_rootNode
        node.id = categorical.id
    
    @staticmethod
    def _unwrap_value(value):
        # If the value was defined in the module numpy, convert it to a
        # Python primitive type for serialization.
        if type(value).__module__ == np.__name__:
            return value.item()
        return value



class BinaryDeserializer:
    """Interface to de-serialize (read) SPNs from SPFlow from an efficient binary format."""

    def __init__(self, fileName):
        """Initialize the de-serializer."""
        self.fileName = fileName

    def deserialize_from_file(self):
        """Deserialize all SPN graphs from the file. Returns a list of SPN graph root nodes."""
        with open(self.fileName, "rb") as inFile:
            # Read header message first
            header = spflow_capnp.Header.read(inFile)
            print(header.which())
            if header.which() == "query":
                return self._deserialize_query(header.query)
            elif header.which() == "model":
                return self._deserialize_model(header.model)
            else:
                raise NotImplementedError(f"No deserialization defined for {header.content}")

    def _deserialize_query(self, query):
        batchSize = query.batchSize
        maxError = query.maxError
        if query.errorKind == "absolute":
            errorKind = ErrorKind.ABSOLUTE
        elif query.errorKind == "relative":
            errorKind = ErrorKind.RELATIVE
        else:
            raise NotImplementedError(f"Cannot deserialize error kind {query.errorKind}")
        errorModel = ErrorModel(errorKind, maxError)
        model = self._deserialize_model(query.joint.model)
        supportsMarginal = query.joint.supportMarginal
        return JointProbability(model, batchSize, supportsMarginal, errorModel)

    def _deserialize_model(self, model):
        rootID = model.rootNode
        featureType = model.featureType
        name = model.name
        if name == "":
            name = None
        rootNodes = self._binary_deserialize_graph(model.nodes)
        for root in rootNodes:
            rebuild_scopes_bottom_up(root)
            assert is_valid(root), "SPN invalid after deserialization"

        rootNode = next((root for root in rootNodes if root.id == rootID), None)
        if rootNode is None:
            logger.error(f"Did not find serialized root node {rootID}")
        return SPNModel(rootNode, featureType, name)


    def _binary_deserialize_graph(self, nodeList):
        node_map = {}
        nodes = []
        for node in nodeList:
            which = node.which()
            deserialized = None
            if which == "product":
                deserialized = self._deserialize_product(node, node_map)
            elif which == "sum":
                deserialized = self._deserialize_sum(node, node_map)
            elif which == "hist":
                deserialized = self._deserialize_histogram(node, node_map)
            elif which == "gaussian":
                deserialized = self._deserialize_gaussian(node, node_map)
            elif which == "categorical":
                deserialized = self._deserialize_categorical(node, node_map)
            else:
                raise NotImplementedError(f"No deserialization defined for {which}")
            node_map[node.id] = deserialized
            if node.rootNode:
                nodes.append(deserialized)
        print(f"Deserialized {len(node_map)} nodes")
        return nodes

    def _deserialize_product(self, node, node_map):
        child_ids = node.product.children
        # Resolve references to child nodes by ID.
        children = [node_map.get(id) for id in child_ids]
        # Check all childs have been resolved.
        assert None not in children, "Child node ID could not be resolved"
        product = Product(children = children)
        product.id = node.id
        return product

    def _deserialize_sum(self, node, node_map):
        child_ids = node.sum.children
        # Resolve references to child nodes by ID.
        children = [node_map.get(id) for id in child_ids]
        # Check all childs have been resolved.
        assert None not in children, "Child node ID could not be resolved"
        sum = Sum(children = children, weights=node.sum.weights)
        sum.id = node.id
        return sum

    def _deserialize_histogram(self, node, node_map):
        breaks = node.hist.breaks
        densities = node.hist.densities
        reprPoints = node.hist.binReprPoints
        type = enum2Type.get(node.hist.type)
        metaType = enum2MetaType.get(node.hist.metaType)
        hist = Histogram(breaks=breaks, densities=densities, bin_repr_points=reprPoints, scope=node.hist.scope,
                        type_=type, meta_type=metaType)
        hist.id = node.id
        return hist

    def _deserialize_gaussian(self, node, node_map):
        gauss = Gaussian(node.gaussian.mean, node.gaussian.stddev, node.gaussian.scope)
        gauss.id = node.id
        return gauss

    def _deserialize_categorical(self, node, node_map):
        probabilities = node.categorical.probabilities
        cat = Categorical(p=probabilities, scope=node.categorical.scope)
        cat.id = node.id
        return cat
