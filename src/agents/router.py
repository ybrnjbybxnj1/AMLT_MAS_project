import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from ..state import AgentState, QueryClassification
from ..llm_utils import invoke_with_parser
from ..agents.memory_manager import get_memory_store


ROUTER_SYSTEM_PROMPT = """
You are a Router Agent in a multi-agent research hypothesis system.

Your role is to analyze user queries and classify them to route to the appropriate specialist agents.

Query Types:
- conceptual: Theoretical questions about MAS, LLM agents, research concepts
- design: Architecture questions, hypothesis design, methodology questions
- implementation: Code, experiments, practical implementation questions
- planning: Full workflow requests needing all specialists (generate hypothesis, design experiment, etc.)

Target Agents:
- research_analyst: For literature review, trend analysis, gap identification
- hypothesis_generator: For hypothesis creation, TRIZ methodology, novelty assessment
- experiment_designer: For experiment design, feasibility, resource planning

You must respond with valid JSON only."""


def router_node(state: AgentState, llm) -> dict:
    # router agent node - classifies query and decides routing
    query = state["user_query"]
    mem = get_memory_store()
    ctx = mem.context(query)
    query_parser = PydanticOutputParser(pydantic_object=QueryClassification)
    prompt = f"""
    Classify this research query.

    Query: {query}
    {f'Previous context: {ctx}' if ctx else 'No previous context.'}
    
    Query types:
    - conceptual: Theory questions, concepts, comparisons
    - design: Architecture, hypothesis design, methodology  
    - implementation: Code, practical how-to, technical details
    - planning: Full research workflow, complete plans
    
    Determine the query type, confidence, whether memory is needed, if it's a follow-up, and which agents should handle it.
    """
    try:
        classification = invoke_with_parser(llm, query_parser, prompt)
        print(f"[ROUTER] {classification.query_type} -> {classification.target_agents}")
        return {
            "classification": classification,
            "agents_activated": ["router"],
            "messages": [f"Router: {classification.query_type}"]
        }
    except Exception as e:
        print(f"[ROUTER] parse failed after retries: {e}, using fallback")
        fallback = QueryClassification(
            query_type="planning",
            confidence="low",
            reasoning="Fallback due to parse error",
            needs_memory=bool(ctx),
            is_followup=False,
            target_agents=["research_analyst", "hypothesis_generator", "experiment_designer"]
        )
        return {
            "classification": fallback,
            "agents_activated": ["router"],
            "messages": ["Router: fallback"]
        }


class RouterAgent:
    # router agent class wrapper
    def __init__(self, llm):
        self.llm = llm
        self.name = "router"
    def __call__(self, state: AgentState) -> dict:
        return router_node(state, self.llm)
