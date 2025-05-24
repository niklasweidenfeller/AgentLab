from dataclasses import asdict, dataclass
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags, MainPrompt
from agentlab.llm.base_api import BaseModelArgs
from agentlab.agents import dynamic_prompting as dp

from time import sleep

from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from browsergym.experiments.agent import AgentInfo

@dataclass
class GenericAgentArgsWithSleep(GenericAgentArgs):
    sleep_time: int = 0

    def make_agent(self):
        return GenericAgentWithSleepAndExtractedMainPrompt(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry, sleep_time=self.sleep_time
        )

class GenericAgentWithSleepAndExtractedMainPrompt(GenericAgent):
    """
    Overrides the GenericAgent to include a sleep time before getting the action.
    This is useful to avoid hitting the rate limit of the AI core.
    
    It also extracts the get_main_prompt method to allow subclasses to override it if needed.
    
    Without any overrides, it behaves like the GenericAgent, but with a sleep time before calling the implementation.
    """

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
        sleep_time: int = 0,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        self.sleep_time = sleep_time

    def get_action(self, obs):

        """
        Get the action from the agent, with a sleep time to avoid hitting the rate limit of the AI core.
        This method overrides the base class method to include a sleep time before calling the implementation.
        """

        print(f"Sleeping for {self.sleep_time} seconds to avoid AI core rate limit")
        sleep(self.sleep_time)
        return self._get_action_impl(obs)

    def get_main_prompt(self) -> MainPrompt:
        return MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

    @cost_tracker_decorator
    def _get_action_impl(self, obs):
        """
        Override the get_action method by extracting a get_main_prompt method,
        which we can override in subclasses if needed.
        """

        self.obs_history.append(obs)
        main_prompt = self.get_main_prompt()

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
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
