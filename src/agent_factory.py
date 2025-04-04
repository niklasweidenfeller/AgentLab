from graph_grounding.agent_args import GraphGroundingAgentArgs, GraphGroundingObsFlags
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

def create_agent_args(use_graph: bool = True, model_name: str = "openai/gpt-4o") -> list:
    flags = GenericPromptFlags(
        obs=GraphGroundingObsFlags(
            use_html=False,
            use_ax_tree=True,
            use_focused_element=True,
            use_error_logs=True,
            use_history=True,
            use_past_error_logs=False,
            use_action_history=True,
            # gpt-4o config except for this line
            use_think_history=True,
            use_diff=False,
            html_type="pruned_html",
            use_screenshot=False,
            use_som=False,
            extract_visible_tag=True,
            extract_clickable_tag=True,
            extract_coords="False",
            filter_visible_elements_only=False,

            ###### GRAPH GROUNDING FLAGS ######
            use_graph=use_graph,
            ###################################
        ),
        action=dp.ActionFlags(
            multi_actions=False,
            action_set="bid",
            long_description=False,
            individual_examples=False,
        ),
        use_plan=False,
        use_criticise=False,
        use_thinking=True,
        use_memory=False,
        use_concrete_example=True,
        use_abstract_example=True,
        use_hints=True,
        enable_chat=False,
        max_prompt_tokens=40_000,
        be_cautious=True,
        extra_instructions=None,
    )

    generic_agent_args = GraphGroundingAgentArgs(
        chat_model_args=CHAT_MODEL_ARGS_DICT[model_name],
        flags=flags,
    )
    return [generic_agent_args]
