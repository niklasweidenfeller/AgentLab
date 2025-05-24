from enum import Enum

class POCIterations(Enum):
    """
    Number of iterations for the POC
    """
    ONE = "standard"
    TWO = "advanced"
    THREE = "llm-generated-graph"
    FOUR = "llm-augmented-graph"
