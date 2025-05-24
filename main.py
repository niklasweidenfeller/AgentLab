"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""
from enum import Enum
from dotenv import load_dotenv
load_dotenv()
import logging
from agentlab.experiments.study import Study
from agent_factory import create_agent_args as create_graph_grounded_agent_args, get_reference_agent_args
from enums import Benchmark, Backend

logging.getLogger().setLevel(logging.INFO)


class Agents(Enum):
    GENERIC_AGENT = "generic_agent"
    GRAPH_GROUNDING_AGENT = "graph_grounding_agent"    

agent = Agents.GENERIC_AGENT

# choose your agent or provide a new agent
agent_args = [
    create_graph_grounded_agent_args(use_graph=True, model_name="aicore/gpt-4o")
    if agent == Agents.GRAPH_GROUNDING_AGENT
    else get_reference_agent_args(model_name="aicore/gpt-4o")
]

# ## select the benchmark to run on
benchmark = Benchmark.WEBARENA.value

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

## Number of parallel jobs
n_jobs = 6  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores


if __name__ == "__main__":  # necessary for dask backend

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.WARNING)

    study.run(
        n_jobs=n_jobs,
        parallel_backend=Backend.RAY.value,
        strict_reproducibility=reproducibility_mode,
        n_relaunch=1,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
