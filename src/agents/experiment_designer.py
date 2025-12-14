import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from ..state import AgentState, ExperimentPlan
from ..tools.experiment_tools import calculate_feasibility, estimate_duration
from ..llm_utils import invoke_with_parser


DESIGNER_SYSTEM_PROMPT = """
You are an Experiment Designer Agent specializing in research experiment planning.

Your responsibilities:
1. Design concrete experiments to test hypotheses
2. Assess feasibility of proposed experiments
3. Estimate required resources and duration
4. Identify potential challenges and mitigation strategies

When designing experiments:
- Be practical and specific
- Consider available resources
- Plan for 3-7 concrete steps
- Account for potential failures

Provide actionable, realistic experiment plans."""


def experiment_designer_node(state: AgentState, llm) -> dict:
    # experiment designer agent node that creates experiment plans
    q = state["user_query"]
    h = state.get("hypothesis")
    print(f"[EXPERIMENT] designing...")
    experiment_parser = PydanticOutputParser(pydantic_object=ExperimentPlan)
    hs = h.statement if h else f"Test the approach: {q}"
    exp_prompt = f"""
    Design an experiment to test this hypothesis.

    Hypothesis: {hs}
    Research context: {q}

    Create a practical experiment plan with:
    - Feasibility assessment (high/medium/low)
    - 3-7 exact steps
    - Required resources
    - Estimated duration
    - Potential challenges
    """
    try:
        experiment = invoke_with_parser(llm, experiment_parser, exp_prompt)
        feas = calculate_feasibility(experiment)
        print(f"[EXPERIMENT] {len(experiment.steps)} steps, feasibility: {feas['category']}")
        return {
            "experiment_plan": experiment,
            "feasibility_score": feas,
            "agents_activated": ["experiment_designer"],
            "messages": [f"Experiment: {len(experiment.steps)} steps"]
        }
    except Exception as e:
        print(f"[EXPERIMENT] parse failed: {e}")
        fallback = ExperimentPlan(
            feasibility="medium",
            steps=["Define experimental setup", "Prepare datasets", "Implement approach", "Run experiments", "Analyze results"],
            resources=["Computing resources", "Datasets", "Evaluation metrics"],
            duration="4-6 weeks",
            challenges=["Data availability", "Computational constraints"]
        )
        return {
            "experiment_plan": fallback,
            "feasibility_score": {"category": "medium", "score": 7},
            "agents_activated": ["experiment_designer"],
            "messages": ["Experiment: fallback"]
        }


class ExperimentDesignerAgent:
    # experiment designer agent class wrapper
    def __init__(self, llm):
        self.llm = llm
        self.name = "experiment_designer"
    def __call__(self, state: AgentState) -> dict:
        return experiment_designer_node(state, self.llm)
