"""
GenericAgent implementation for AgentLab

This module defines a `GenericAgent` class and its associated arguments for use in the AgentLab framework. \
The `GenericAgent` class is designed to interact with a chat-based model to determine actions based on \
observations. It includes methods for preprocessing observations, generating actions, and managing internal \
state such as plans, memories, and thoughts. The `GenericAgentArgs` class provides configuration options for \
the agent, including model arguments and flags for various behaviors.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from warnings import warn

import bgym
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs, make_system_message, make_user_message
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from .generic_agent_prompt import GenericPromptFlags, MainPrompt


@dataclass
class GenericAgentArgs(AgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: GenericPromptFlags = None
    max_retry: int = 4

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"GenericAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        """Override Some flags based on the benchmark."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(benchmark.high_level_action_set_args)

        # for backward compatibility with old traces
        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict

        # verify if we can remove this
        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        return GenericAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )

import os
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

class GenericAgent(Agent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
    ):
        self.navigation_graph = MemgraphNavigationGraph("bolt://localhost:7687", AUTH)

        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry

        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)

        self._check_flag_constancy()
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        updated_obs = self._obs_preprocessor(obs)
        if not self.flags.obs.use_graph:
            return updated_obs

        #######################################################
        # Inference Time: Query the navigation graph to get   #
        # possible next actions to take.                      #
        #######################################################
        url = build_abstract_url(updated_obs["url"])
        url = replace_urls(url)

        try:
            current_node_id = self.navigation_graph.get_page_node_id(url)
            updated_obs["graph_grounding"] = self.navigation_graph.infer_actions_at_position(current_node_id, n_hops=2) if current_node_id else None
        except ValueError:
            print(f"Could not find page node for url: {url}")
            updated_obs["graph_grounding"] = None
        return updated_obs

    @cost_tracker_decorator
    def get_action(self, obs):

        self.obs_history.append(obs)
        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )

        print("***************")
        print(system_prompt)
        print("***************")
        print(human_prompt)
        print("***************")

        try:
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long

            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []

    def _check_flag_constancy(self):
        flags = self.flags
        if flags.obs.use_som:
            if not flags.obs.use_screenshot:
                warn(
                    """
Warning: use_som=True requires use_screenshot=True. Disabling use_som."""
                )
                flags.obs.use_som = False
        if flags.obs.use_screenshot:
            if not self.chat_model_args.vision_support:
                warn(
                    """
Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                flags.obs.use_screenshot = False
        return flags

    def _get_maxes(self):
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunc_itr


from functools import partial


def get_action_post_hoc(agent: GenericAgent, obs: dict, ans_dict: dict):
    """
    Get the action post-hoc for the agent.

    This function is used to get the action after the agent has already been run.
    Its goal is to recreate the prompt and the output of the agent a posteriori.
    The purpose is to build datasets for training the agents.

    Args:
        agent (GenericAgent): The agent for which the action is being determined.
        obs (dict): The observation dictionary to append to the agent's history.
        ans_dict (dict): The answer dictionary containing the plan, step, memory, think, and action.

    Returns:
        Tuple[str, str]: The complete prompt used for the agent and the reconstructed output based on the answer dictionary.
    """
    system_prompt = dp.SystemPrompt().prompt

    agent.obs_history.append(obs)

    main_prompt = MainPrompt(
        action_set=agent.action_set,
        obs_history=agent.obs_history,
        actions=agent.actions,
        memories=agent.memories,
        thoughts=agent.thoughts,
        previous_plan=agent.plan,
        step=agent.plan_step,
        flags=agent.flags,
    )

    max_prompt_tokens, max_trunc_itr = agent._get_maxes()

    fit_function = partial(
        dp.fit_tokens,
        max_prompt_tokens=max_prompt_tokens,
        model_name=agent.chat_model_args.model_name,
        max_iterations=max_trunc_itr,
    )

    instruction_prompt = fit_function(shrinkable=main_prompt)

    if isinstance(instruction_prompt, list):
        # NOTE: this is when we have images
        instruction_prompt = instruction_prompt[0]["text"]

    # TODO: make sure the bid is in the prompt

    output = ""

    # TODO: validate this
    agent.plan = ans_dict.get("plan", agent.plan)
    if agent.plan != "No plan yet":
        output += f"\n<plan>\n{agent.plan}\n</plan>\n"

    # TODO: is plan_step something that the agent's outputs?
    agent.plan_step = ans_dict.get("step", agent.plan_step)

    memory = ans_dict.get("memory", None)
    agent.memories.append(memory)
    if memory is not None:
        output += f"\n<memory>\n{memory}\n</memory>\n"

    thought = ans_dict.get("think", None)
    agent.thoughts.append(thought)
    if thought is not None:
        output += f"\n<think>\n{thought}\n</think>\n"

    action = ans_dict["action"]
    agent.actions.append(action)
    if action is not None:
        output += f"\n<action>\n{action}\n</action>"

    return system_prompt, instruction_prompt, output


from urllib.parse import urlparse
from dataclasses import dataclass
from neo4j import Driver, GraphDatabase

@dataclass
class Webpage:
    """ A class to represent a webpage. """
    id: int | None
    abstract_url: str

class WebpageRepository:
    def __init__(self, client: Driver):
        self.client = client

    def find_by_url(self, abstract_url: str) -> Webpage:
        records, _, _ = self.client.execute_query(
            "MATCH (a:URL) WHERE a.url = $url RETURN elementID(a), a.url",
            url=abstract_url
        )
        if len(records) == 0:
            raise ValueError(f'No webpage found with abstract URL: {abstract_url}')
        return Webpage(id=records[0][0], abstract_url=records[0][1])

class MemgraphNavigationGraph:
    def __init__(self, uri: str, auth: tuple[str, str]):
        self.client = GraphDatabase.driver(uri, auth=auth)
        self.client.verify_connectivity()
        self.webpage_repository = WebpageRepository(self.client)
    
    def __del__(self):
        self.client.close()

    """
    RAG-based inference at runtime
    """
    def get_page_node_id(self, url: str) -> int:
        return self.webpage_repository.find_by_url(url).id

    def infer_actions_at_position(self, node_id: int, *, n_hops: int = 1):
        return self.query_longest_paths(node_id)

    def query_longest_paths(self, source_id: int):

        MAX_PATH_LENGTH = 10

        query = f'''
        MATCH (a:URL), (b:URL)
        WHERE elementID(a) = $source_id AND a <> b
        CALL apoc.algo.allSimplePaths(a, b, 'ACTION', {MAX_PATH_LENGTH})
        YIELD path

        LIMIT 200

        WITH COLLECT(path) as paths

        CALL navgraph.getPrimePaths(paths)
        YIELD primePath as path
        WITH COLLECT(path) as paths
        CALL navgraph.allNodesInPathHaveSameValueInNodePropertyArray(paths, 'flows')
        YIELD path

        WITH
            [node in nodes(path) | node.url] as nodesInPath,
            [rel in relationships(path) | rel.action + ": " + rel.target_line] as actionDescriptions, 
            length(path) as pathLength
        ORDER BY pathLength DESC
        WITH
            REDUCE(acc = [], i IN RANGE(0, pathLength - 2) |
                acc + [nodesInPath[i], actionDescriptions[i]]
            ) + [LAST(nodesInPath)] as nodeEdgeArray
        RETURN nodeEdgeArray
        '''

        print("******Querying known paths ...*")
        print(f"******Source ID: {source_id} ...*")
              
        records, _, _ = self.client.execute_query(
            query,
            source_id=source_id
        )

        paths = [r['nodeEdgeArray'] for r in records]
        merged = [ self.merge_path(path) for path in paths ]
        merged = [ path for path in merged if path ]
        print("******Number of paths found: ", len(merged))

        text = "\n\n".join([f"\t{i+1}. {path}" for i, path in enumerate(merged)]).strip()
        return text if text else None

    def merge_path(self, path):
        if len(path) == 1:
            print("Path length is 1")
            return None
        
        parts = path
        cleaned_parts = []

        current_url = None

        for part in parts:
            if not current_url:
                current_url = self.remove_host(part)
            else:
                cleaned_parts.append((current_url, part))
                current_url = None

        joined_tuples = [ f"({tup[0]}, {tup[1]})" for tup in cleaned_parts ]

        # append the end state
        if len(parts) % 2 != 0:
            joined_tuples.append(f"({self.remove_host(parts[-1])})")

        return " -> ".join(joined_tuples)

    def remove_host(self, url: str) -> str:
        parsed_url = urlparse(url)

        cleaned = parsed_url.path
        if parsed_url.query:
            cleaned += "?" + parsed_url.query
        if parsed_url.fragment:
            cleaned += "#" + parsed_url.fragment    
        return cleaned

def build_abstract_url(url: str) -> str:
    # first, we need to parse the URL
    parsed_url = urlparse(url)
    query = parsed_url.query

    # convert the query into a dictionary
    query_parts = query.split("&")
    query_dict = {}

    for part in query_parts:
        if (part == ""):
            continue
        split_parts = part.split("=")
        if (len(split_parts) == 0):
            continue

        if (len(split_parts) == 1):
            key, value = part, ""
        else:
            key = split_parts[0]
            value = "=".join(split_parts[1:])
            query_dict[key] = value

    # replace the values with placeholders
    for key in query_dict:
        query_dict[key] = f"<{key}>"

    # convert the dictionary back into a query string
    query = "&".join([f"{key}={value}" for key, value in query_dict.items()]) if len(query_dict.items()) > 0 else ""
    fragment = "#<fragment>" if parsed_url.fragment else ""

    # rebuild the URL
    abstract_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}{query}{fragment}"
    return abstract_url
_to_replace = [
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:3000", "https://osm.org"),
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:7770", "https://shopping.com"),
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:7780", "https://shopping.com"),
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:8023", "https://vcs.com"),
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:9980", "https://marketplace.com"),
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:9999", "https://social-forum.com"),
    ("https://dev275528.service-now.com", "https://servicenow.test"),
]

def replace_urls(url, to_replace = _to_replace):
    for old, new in to_replace:
        url = url.replace(old, new)
    return url

