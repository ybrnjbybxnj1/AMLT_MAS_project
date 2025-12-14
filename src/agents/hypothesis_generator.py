import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from ..state import AgentState, Hypothesis
from ..tools.hypothesis_tools import calculate_novelty, TRIZ_PRINCIPLES
from ..llm_utils import invoke_with_parser


GENERATOR_SYSTEM_PROMPT = """
You are a Hypothesis Generator Agent specializing in creative research hypothesis formulation.

Your responsibilities:
1. Generate novel research hypotheses using TRIZ methodology
2. Assess hypothesis novelty and originality
3. Provide clear rationale for hypotheses
4. Ensure hypotheses are specific and testable

TRIZ (Theory of Inventive Problem Solving) principles you can apply:
- Segmentation, Merging, Universality
- Asymmetry, Nested doll, Anti-weight
- Preliminary action, Dynamics, Another dimension
- Feedback, Self-service, Copying

Generate hypotheses that are creative yet grounded in research evidence.
"""


def hypothesis_generator_node(state: AgentState, llm) -> dict:
    # hypothesis generator agent node that creates research hypotheses
    q = state["user_query"]
    tr = state.get("trends")
    ga = state.get("gaps")
    papers = state.get("papers", [])
    print(f"[HYPOTHESIS] generating...")
    hypothesis_parser = PydanticOutputParser(pydantic_object=Hypothesis)
    tl = (tr.trends if tr else ["Emerging research"])[:3]
    gl = (ga.opportunities if ga else ["Novel approaches"])[:3]
    hyp_prompt = f"""
    Generate a research hypothesis using TRIZ methodology.

    Research question: {q}

    Current trends: {', '.join(tl)}
    Research opportunities: {', '.join(gl)}

    TRIZ Principles to consider: {', '.join(TRIZ_PRINCIPLES[:5])}

    Create a specific, testable hypothesis with:
    - A clear statement (minimum 20 characters)
    - Which TRIZ principles apply
    - Rationale for why this hypothesis matters
    - Novelty score (1-10)
    """
    try:
        hypothesis = invoke_with_parser(llm, hypothesis_parser, hyp_prompt)
        nv = calculate_novelty(hypothesis.statement, papers)
        print(f"[HYPOTHESIS] created, novelty: {nv['score']}/10")
        return {
            "hypothesis": hypothesis,
            "novelty_score": nv,
            "agents_activated": ["hypothesis_generator"],
            "messages": [f"Hypothesis: novelty {nv['score']}"]
        }
    except Exception as e:
        print(f"[HYPOTHESIS] parse failed: {e}")
        fallback = Hypothesis(
            statement=f"Applying {TRIZ_PRINCIPLES[0]} principle to {q} will improve research outcomes",
            triz_principles=TRIZ_PRINCIPLES[:2],
            rationale="Addresses identified research gaps through systematic innovation",
            novelty_score=6
        )
        return {
            "hypothesis": fallback,
            "novelty_score": {"score": 6},
            "agents_activated": ["hypothesis_generator"],
            "messages": ["Hypothesis: fallback"]
        }


class HypothesisGeneratorAgent:
    # hypothesis generator agent class wrapper
    def __init__(self, llm):
        self.llm = llm
        self.name = "hypothesis_generator"
    def __call__(self, state: AgentState) -> dict:
        return hypothesis_generator_node(state, self.llm)
