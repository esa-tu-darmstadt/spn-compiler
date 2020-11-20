from xspn.structure.Model import SPNModel


class Query:
    """Abstract super class of the different types of queries that can be 
    performed on an SPN. 

    Args:
        batchSize (int): Batch size to optimize for, or 1 to optimize for single evaluation.

    Attributes:
        batchSize (int): Batch size to optimize for, or 1 to optimize for single evaluation.

    """

    def __init__(self, batchSize):
        self.batchSize = batchSize



class JointProbability(Query):
    """Query SPN for joint probability, with full evidence

    Args: 
        model (SPNModel): Single SPN graph to run query on.
        batchSize (int): Batch size to optimize for, or 1 to optimize for single evaluation.
        rootNodeRelativeError (float): Maximum relative error allowed at the root node.

    Attributes: 
        graph (SPNModel): Single SPN graph to run query on.
        rootError (float): Maximum relative error allowed at the root node.   

    """

    def __init__(self, model : SPNModel, batchSize = 1, rootNodeRelativeError = 0.02):
        Query.__init__(self, batchSize)
        self.graph = model
        self.rootError = rootNodeRelativeError

    def models(self):
        return [self.graph]