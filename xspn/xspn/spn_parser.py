from lark import Lark, Transformer, Token
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.Base import Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, get_nodes_by_type
import itertools
import numpy as np


# Supports the variable list at the bottom of the file
parser = Lark(
"""

%import common._STRING_ESC_INNER -> STRING
%import common.SIGNED_NUMBER -> NUMBER
%ignore WHITESPACE

WHITESPACE : " " | "\\n" | "\\t"

ID : ("a".."z" | "A".."Z" | "0".."9" | "_")+

start : node varlist
node : sumnode | productnode | histogramnode

sumnode : ID "SumNode" "(" sumargs ")" "{" nodebody "}"
sumargs : (NUMBER "*" ID)? ("," NUMBER "*" ID)*

productnode : ID "ProductNode" "(" productargs ")" "{" nodebody "}"
productargs: ID? ("," ID)*

nodebody : (node)*

histogramnode : ID "Histogram" "(" ID "|" numberarray ";" numberarray [";" numberarray] ")"

numberarray : "[" NUMBER? ("," NUMBER)* "]"

varlist: "#" ID (";" ID)*

""")


class HistogramNode:
    name = ""
    variable_name = ""
    breaks = []
    probabilities = []

    def __init__(self, name, variable_name, breaks, probabilities):
        self.name = name
        self.variable_name = variable_name
        self.breaks = breaks
        self.probabilities = probabilities

    def __repr__(self):
        return (f"Histogram {self.name} | {self.variable_name}\n" +
                str(self.breaks) + "\n" +
                str(self.probabilities))

    def to_spn(self, variables_to_index):
        #diffs = [int(np.ceil(b - a)) for a, b in zip(self.breaks, self.breaks[1:])]
        #assert(len(self.probabilities) == len(diffs))
        #probs = list(itertools.chain(*[[p] * d  for p, d in zip(self.probabilities, diffs)]))
        #norm = sum(probs)
        # TODO: Why do the probabilities normally not add up to 1?
        #probs = [p / norm for p in probs]
        #return Categorical(p=probs, scope=variables_to_index[self.variable_name])
        #print(self.breaks)
        return Histogram(breaks=self.breaks,
                         densities=self.probabilities,
                         bin_repr_points=self.breaks,
                         scope=[variables_to_index[self.variable_name]]
                         )


class ProductNode:
    name = ""
    children = []

    def __init__(self, name, children):
        self.name = name
        self.children = children

    def __repr__(self):
        return (f"Product {self.name}\n" +
                str(self.children))

    def to_spn(self, variables_to_index):
        return Product(children=[child.to_spn(variables_to_index) for child in self.children])


class SumNode:
    name = ""
    children = []
    weights = []

    def __init__(self, name, children, weights):
        self.name = name
        self.children = children
        self.weights = weights

        assert(len(children) == len(weights))

    def __repr__(self):
        return (f"Sum {self.name}\n" +
                str(self.weights) + "\n" +
                str(self.children))

    def to_spn(self, variables_to_index):
        c = [child.to_spn(variables_to_index) for child in self.children]
        return Sum(weights=self.weights, children=c)


class TreeToSPN(Transformer):
    def __init__(self):
        self.histograms = dict()
        pass

    def start(self, s):
        return (s[1], s[0])

    def node(self, s):
        return s[0]

    def varlist(self, s):
        return s

    def nodebody(self, s):
        return s

    def sumnode(self, s):
        name = s[0]

        # TODO: Maybe we shouldn't assume the same order for weights and children!
        weights = [w for i, w in enumerate(s[1]) if i % 2 == 0]
        children_names = [w for i, w in enumerate(s[1]) if i % 2 == 1]

        children = s[2]

        return SumNode(name, children, weights)

    def sumargs(self, s):
        return s

    def productnode(self, s):
        name = s[0]
        children_names = s[1]
        children = s[2]

        return ProductNode(name, children)

    def productargs(self, s):
        return s

    def histogramnode(self, s):
        name = s[0]
        var_id = s[1]
        classes = s[2]
        densities = s[3]

        return HistogramNode(s[0], s[1], s[2], s[3])

    def ID(self, s):
        return str(s)

    def numberarray(self, s):
        return s

    def NUMBER(self, s):
        return float(s)


def tree_to_spn(parse_tree):
    pass


def load_spn(path):
    text = open(path, "r").read()
    parse_tree = parser.parse(text)
    variables, tree = TreeToSPN().transform(parse_tree)
    variables_to_index = dict([(b, a) for a, b in enumerate(variables)])

    spn = tree.to_spn(variables_to_index)

    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)

    # for each variable get one histogram on that variable and compute the min/max input value
    histogram_nodes = get_nodes_by_type(spn, Histogram)
    index_to_min = dict()
    index_to_max = dict()

    for var, index in variables_to_index.items():
        # find any histogram
        for histo in histogram_nodes:
            if index == histo.scope[0]:
                index_to_min[index] = int(min(histo.breaks))
                index_to_max[index] = int(max(histo.breaks))
                break


    return spn, variables_to_index, index_to_min, index_to_max


def load_spn_2(path):
    grammar = (
    r"""
%import common.DECIMAL -> DECIMAL
%import common.SIGNED_NUMBER -> NUMBERS
%import common.WS
%ignore WS
%import common.WORD -> WORD
%import common.DIGIT -> DIGIT
ALPHANUM: "a".."z"|"A".."Z"|DIGIT
PARAMCHARS: ALPHANUM|"_"
FNAME: ALPHANUM+
PARAMNAME: PARAMCHARS+
list: "[" [NUMBERS ("," NUMBERS)*] "]"


?spn: [node varlist]
?node: prodnode | sumnode | histonode
"""
        + r"""

prodnode: [PARAMNAME] "ProductNode" "(" [PARAMNAME ("," PARAMNAME)*] ")" "{" [node*] "}"
sumnode: [PARAMNAME] "SumNode" "(" [NUMBERS "*" PARAMNAME ("," NUMBERS "*" PARAMNAME)*] ")" "{" [node*] "}"
histonode: [PARAMNAME] "Histogram" "(" [PARAMNAME] "|" list ";" list ")"
varlist: ["#" PARAMNAME (";" PARAMNAME)*]

"""
    )

    with open(path, 'r') as file:
        text = file.read()

    parse_tree = Lark(grammar, start="spn").parse(text)

    def print_tree(node, spaces = ''):
        print(f'{spaces}{node.data}')
        for child in node.children:
            print_tree(child, spaces + '    ')

    #print_tree(parse_tree)
    #exit()

    def partition(list, pred):
        yes = [e for e in list if pred(e)]
        no = [e for e in list if not pred(e)]
        return yes, no

    def process_var_list(tree):
        return [str(child) for child in tree.children]

    def tree_to_spn(tree, var_to_index):
        tnode = tree.data

        if tnode == 'spn':
            var_list = process_var_list(tree.children[1])
            v2i = {v: i for i, v in enumerate(var_list)}
            return var_list, tree_to_spn(tree.children[0], v2i)

        if tnode == "sumnode":
            node = Sum()

            tokens, children = partition(tree.children, lambda x: isinstance(x, Token))
            # Unused?
            node_name = str(tokens[0])
            weights = [float(tok) for tok in tokens[1::2]]
            children = [tree_to_spn(child, var_to_index) for child in children]

            return Sum(weights, children)

        if tnode == "prodnode":
            _, children = partition(tree.children, lambda x: isinstance(x, Token))
            children = [tree_to_spn(child, var_to_index) for child in children]

            return Product(children)

        if tnode == "histonode":
            tokens, children = partition(tree.children, lambda x: isinstance(x, Token))

            node_name = str(tokens[0])
            var_name = str(tokens[1])
            breaks_list = [int(e) for e in tree_to_spn(children[0], var_to_index)]
            probs_list = tree_to_spn(children[1], var_to_index)

            return Histogram(breaks_list, probs_list, breaks_list, scope=[var_to_index[var_name]])

        if tnode == 'list':
            return [float(child) for child in tree.children]

        raise Exception("Node type not registered: " + tnode)

    var_list, spn = tree_to_spn(parse_tree, None)
    assign_ids(spn)
    spn = rebuild_scopes_bottom_up(spn)
    var_2_index = {v: i for i, v in enumerate(var_list)}

    return spn, var_2_index, [], []