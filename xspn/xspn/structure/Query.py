from xspn.structure.Model import SPNModel
from enum import IntEnum

class ErrorKind(IntEnum):
    """Enumeration of different ways to compute arithmetic error during error analysis.

    ABSOLUTE is the absolute error: delta_f = |f' - f|
    RELATIVE is the relative error: relative_f = |f' - f|/f

    """

    ABSOLUTE = 1
    RELATIVE = 2

class ErrorModel:

    """Represents an error model for arithmetic error analysis, comprising the error kind
       and maximum value allowed for the error.
    """

    def __init__(self, kind = ErrorKind.RELATIVE, error = 0.02):
        self.kind = kind
        self.error = error


class Query:
    """Abstract super class of the different types of queries that can be 
    performed on an SPN. 

    Args:
        batchSize (int): Batch size to optimize for, or 1 to optimize for single evaluation.
        errorModel (ErrorModel): Error model for arithmetic error analysis of the query.

    Attributes:
        batchSize (int): Batch size to optimize for, or 1 to optimize for single evaluation.
        errorModel (ErrorModel): ErroKind and maximum allowed error for arithmetic error analysis.

    """

    def __init__(self, batchSize, errorModel):
        self.batchSize = batchSize
        self.errorModel = errorModel



class JointProbability(Query):
    """Query SPN for joint probability, with full evidence

    Args: 
        model (SPNModel): Single SPN graph to run query on.
        batchSize (int): Batch size to optimize for, or 1 to optimize for single evaluation.
        rootError (ErrorModel): Maximum relative error allowed at the root node.

    Attributes: 
        graph (SPNModel): Single SPN graph to run query on.

    """

    def __init__(self, model: SPNModel, batchSize=1, supportMarginal=False,
                 rootError=ErrorModel(ErrorKind.ABSOLUTE, 0.02)):
        Query.__init__(self, batchSize, rootError)
        self.graph = model
        self.marginal = supportMarginal

    def models(self):
        return [self.graph]

    def supportsMarginal(self):
        return self.marginal
