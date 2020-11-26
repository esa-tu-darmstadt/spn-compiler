import os

from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import get_number_of_nodes,get_number_of_edges
from xspn.serialization.binary.BinarySerialization import BinarySerializer, BinaryDeserializer
from xspn.structure.Model import SPNModel
from xspn.structure.Query import JointProbability, ErrorKind

def test_binary_serialization_roundtrip(tmpdir):
    """Tests the binary serialization for SPFlow SPNs by round-tripping 
    a simple SPN through serialization and de-serialization and comparing
    the graph-structure before and after serialization & de-serialization."""
    h1 = Histogram([0., 1., 2.], [0.25, 0.75], [1, 1], scope=1)
    h2 = Histogram([0., 1., 2.], [0.45, 0.55], [1, 1], scope=2)
    h3 = Histogram([0., 1., 2.], [0.33, 0.67], [1, 1], scope=1)
    h4 = Histogram([0., 1., 2.], [0.875, 0.125], [1, 1], scope=2)

    p0 = Product(children=[h1, h2])
    p1 = Product(children=[h3, h4])
    spn = Sum([0.3, 0.7], [p0, p1])

    model = SPNModel(spn, featureValueType="uint32")
    query = JointProbability(model)

    binary_file = os.path.join(tmpdir, "test.bin")
    print(f"Test binary file: {binary_file}")

    BinarySerializer(binary_file).serialize_to_file(query)

    deserialized = BinaryDeserializer(binary_file).deserialize_from_file()

    assert(isinstance(deserialized, JointProbability))
    assert(deserialized.batchSize == query.batchSize)
    assert(deserialized.errorModel.error == query.errorModel.error)
    assert(deserialized.errorModel.kind == query.errorModel.kind)
    assert(deserialized.graph.featureType == model.featureType)
    assert(deserialized.graph.name == model.name)

    deserialized = deserialized.graph.root
    assert get_number_of_nodes(spn) == get_number_of_nodes(deserialized)
    assert get_number_of_nodes(spn, Sum) == get_number_of_nodes(deserialized, Sum)
    assert get_number_of_nodes(spn, Product) == get_number_of_nodes(deserialized, Product)
    assert get_number_of_nodes(spn, Histogram) == get_number_of_nodes(deserialized, Histogram)
    assert get_number_of_edges(spn) == get_number_of_edges(deserialized)


def test_categorical_leaf_serialization(tmpdir):
    """Tests the binary serialization of two SPFlow Categorical leaf nodes
    by round-tripping and comparing the parameters before and after serialization
    & deserialization"""
    c1 = Categorical(p=[0.35, 0.55, 0.1], scope=1)
    c2 = Categorical(p=[0.25, 0.625, 0.125], scope=2)
    p = Product(children=[c1, c2])

    binary_file = os.path.join(tmpdir, "test.bin")
    print(f"Test binary file: {binary_file}")

    model = SPNModel(p, "uint8", "test")
    query = JointProbability(model)

    BinarySerializer(binary_file).serialize_to_file(query)

    deserialized = BinaryDeserializer(binary_file).deserialize_from_file()
    
    assert(isinstance(deserialized, JointProbability))
    assert(isinstance(deserialized.graph, SPNModel))
    assert(deserialized.graph.featureType == model.featureType)
    assert(deserialized.graph.name == model.name)
    
    deserialized = deserialized.graph.root

    assert isinstance(deserialized, Product)
    assert(len(deserialized.children) == 2)
    assert len(c1.p) == len(deserialized.children[0].p)
    for i,p in enumerate(c1.p):
        assert p == deserialized.children[0].p[i]
    assert len(c2.p) == len(deserialized.children[1].p)
    for i,p in enumerate(c2.p):
        assert p == deserialized.children[1].p[i]

def test_gaussian_leaf_serialization(tmpdir):
    """Tests the binary serialization of two SPFlow Gaussian leaf nodes
    by round-tripping and comparing the parameters before and after serialization
    & deserialization"""
    g1 = Gaussian(mean=0.5, stdev=1, scope=0)
    g2 = Gaussian(mean=0.125, stdev=0.25, scope=1)
    p = Product(children=[g1, g2])

    binary_file = os.path.join(tmpdir, "test.bin")
    print(f"Test binary file: {binary_file}")

    model = SPNModel(p, "float32", "test")
    query = JointProbability(model)

    BinarySerializer(binary_file).serialize_to_file(query)

    deserialized = BinaryDeserializer(binary_file).deserialize_from_file()

    assert(isinstance(deserialized, JointProbability))
    assert(isinstance(deserialized.graph, SPNModel))
    assert(deserialized.graph.featureType == model.featureType)
    assert(deserialized.graph.name == model.name)

    deserialized = deserialized.graph.root

    assert isinstance(deserialized, Product)
    assert(len(deserialized.children) == 2)
    gaussian1 = deserialized.children[0]
    gaussian2 = deserialized.children[1]
    assert(g1.scope == gaussian1.scope)
    assert(g1.mean == gaussian1.mean)
    assert(g1.stdev == gaussian1.stdev)
    assert(g2.scope == gaussian2.scope)
    assert(g2.mean == gaussian2.mean)
    assert(g2.stdev == gaussian2.stdev)