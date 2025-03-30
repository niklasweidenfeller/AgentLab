from enum import Enum

class Benchmark(Enum):
    """
    Enum class for different benchmarks.
    """
    MINIWOB = "miniwob"
    MINIWOB_TINY_TEST = "miniwob_tiny_test"
    WORKARENA_L1 = "workarena_l1"
    WORKARENA_L2 = "workarena_l2"
    WORKARENA_L3 = "workarena_l3"
    WEBARENA = "webarena"
    VISUALWEBARENA = "visualwebarena"

class Backend(Enum):
    """
    Enum class for different backends.
    """
    DASK = "dask"
    RAY = "ray"
    JOBLIB = "joblib"
    SEQUENTIAL = "sequential"
