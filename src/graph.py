"""
LangGraph Construction for Multi-Agent Hypothesis Planner
"""
from typing import Literal
from langgraph.graph import StateGraph, END

from .state import AgentState, create_initial_state
from .agents.router import RouterAgent
from .agents.research_analyst import ResearchAnalystAgent
from .agents.hypothesis_generator import HypothesisGeneratorAgent
from .agents.experiment_designer import ExperimentDesignerAgent
from .agents.memory_manager import memory_retrieval_node, memory_update_node

# agent name mapping LLM-returned names to actual node names
AGENT_NAME_MAP = {
    # research-related
    "research": "research_analyst", "researcher": "research_analyst", "research_analyst": "research_analyst",
    "research assistant": "research_analyst", "researchassistant": "research_analyst",
    "theory agent": "research_analyst", "theoryagent": "research_analyst",
    "literature": "research_analyst", "analyst": "research_analyst",
    "researchagent": "research_analyst", "theoryanalysis": "research_analyst",
    "researchdesign": "research_analyst", "researchplanningagent": "research_analyst",
    "literaturereview": "research_analyst", "literatureagent": "research_analyst",
    # hypothesis-related
    "hypothesis": "hypothesis_generator", "hypothesisgenerator": "hypothesis_generator",
    "hypothesis_generator": "hypothesis_generator", "hypothesisdesignagent": "hypothesis_generator",
    "design": "hypothesis_generator", "designer": "hypothesis_generator",
    "multiagentsystemsagent": "hypothesis_generator", "multiagentarchitect": "hypothesis_generator",
    "systemarchitecture": "hypothesis_generator", "systemdesignagent": "hypothesis_generator",
    "designagents": "hypothesis_generator", "trizexperts": "hypothesis_generator",
    "trizagent": "hypothesis_generator", "architectureagent": "hypothesis_generator",
    "multiagentsystemarchitect": "hypothesis_generator",
    # experiment-related
    "experiment": "experiment_designer", "experimentdesigner": "experiment_designer",
    "experiment_designer": "experiment_designer", "implementation": "experiment_designer",
    "planner": "experiment_designer", "researchplanner": "experiment_designer",
    "memorysystemdesigner": "experiment_designer", "memorymanagementagent": "experiment_designer",
    "langgraphdeveloperagent": "experiment_designer", "errorhandlingspecialist": "experiment_designer",
    "implementationagent": "experiment_designer", "codeagent": "experiment_designer",
    "developeragent": "experiment_designer", "practicalagent": "experiment_designer",
}

def normalize_agent_name(name: str) -> str:
    # maps LLM-returned agent names to actual node names
    normalized = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    return AGENT_NAME_MAP.get(normalized, None)


def get_target_agents_normalized(target_agents: list) -> set:
    # normalizes a list of target agents to actual node names
    normalized = set()
    for agent in target_agents:
        mapped = normalize_agent_name(agent)
        if mapped:
            normalized.add(mapped)
    return normalized


def create_synthesizer_node(llm):
    # create the final response synthesizer node
    from langchain_core.messages import SystemMessage, HumanMessage
    from .llm_utils import llm_retry
    
    @llm_retry()
    def synth_llm_call(parts):
        # synthesizer LLM call with retry
        return llm.invoke([
            SystemMessage(content="Synthesize multi-agent findings into a clear, comprehensive response."),
            HumanMessage(content="\n".join(parts))
        ]).content.strip()
    
    def synthesizer_node(state: AgentState) -> dict:
        # synthesize final response from all agent outputs 
        q = state["user_query"]
        c = state.get("classification")
        tr, ga = state.get("trends"), state.get("gaps")
        h, e = state.get("hypothesis"), state.get("experiment_plan")
        nv, fv = state.get("novelty_score", {}), state.get("feasibility_score", {})
        print(f"[SYNTH] generating response...")
        parts = [f"Query: {q}"]
        if c:
            parts.append(f"Query type: {c.query_type}")
        if tr:
            parts.append(f"Trends: {', '.join(tr.trends)}")
        if ga:
            parts.append(f"Opportunities: {', '.join(ga.opportunities)}")
        if h:
            parts.append(f"Hypothesis: {h.statement}")
            parts.append(f"TRIZ principles: {', '.join(h.triz_principles)}")
            parts.append(f"Novelty score: {nv.get('score', h.novelty_score)}/10")
        if e:
            parts.append(f"Experiment: {len(e.steps)} steps, Duration: {e.duration}")
            parts.append(f"Feasibility: {fv.get('category', e.feasibility)}")
        try:
            response = synth_llm_call(parts)
        except Exception as ex:
            print(f"[SYNTH] LLM failed after retries: {ex}")
            response = "\n".join(parts)
        print(f"[SYNTH] response: {len(response)} chars")
        return {
            "final_response": response,
            "agents_activated": ["synthesizer"]
        }
    return synthesizer_node


def route_after_router(state: AgentState) -> str:
    # determine next node after router based on classification
    classification = state.get("classification")
    if not classification:
        return "research_analyst"  # Default
    # check if memory is needed first
    if classification.needs_memory:
        return "memory_retrieval"
    # route based on query type
    if classification.query_type == "conceptual":
        return "research_analyst"
    elif classification.query_type == "design":
        return "research_analyst"  # start with research, then hypothesis
    elif classification.query_type == "implementation":
        return "experiment_designer"
    else:  # planning - full workflow
        return "research_analyst"


def route_after_memory(state: AgentState) -> str:
    # route after memory retrieval - uses normalized agent names
    classification = state.get("classification")
    if not classification:
        return "research_analyst"
    normalized_agents = get_target_agents_normalized(classification.target_agents)
    print(f"[ROUTE] target_agents={classification.target_agents} -> normalized={normalized_agents}")
    # check normalized agents
    if "research_analyst" in normalized_agents:
        return "research_analyst"
    elif "hypothesis_generator" in normalized_agents:
        return "hypothesis_generator"
    elif "experiment_designer" in normalized_agents:
        return "experiment_designer"
    # fallback to query_type
    print(f"[ROUTE] fallback to query_type={classification.query_type}")
    if classification.query_type == "conceptual":
        return "research_analyst"
    elif classification.query_type == "design":
        return "research_analyst"  # design needs research first
    elif classification.query_type == "implementation":
        return "experiment_designer"
    else:  # planning
        return "research_analyst"


def route_after_research(state: AgentState) -> str:
    # route after research analyst
    classification = state.get("classification")
    if not classification:
        return "synthesizer"
    # for design/planning queries, go to hypothesis generator
    if classification.query_type in ["design", "planning"]:
        return "hypothesis_generator"
    # for conceptual queries, synthesize directly
    return "synthesizer"


def route_after_hypothesis(state: AgentState) -> str:
    # route after hypothesis generator
    classification = state.get("classification")
    if not classification:
        return "synthesizer"
    # for planning queries, continue to experiment designer
    if classification.query_type == "planning":
        return "experiment_designer"
    return "synthesizer"


def route_after_experiment(state: AgentState) -> str:
    # route after experiment designer
    return "synthesizer"


def route_after_synthesizer(state: AgentState) -> str:
    # route after synthesizer - always update memory then end
    return "memory_update"


def build_graph(llm):
    # build the multi-agent LangGraph
    # initialize agents
    router = RouterAgent(llm)
    research_analyst = ResearchAnalystAgent(llm)
    hypothesis_generator = HypothesisGeneratorAgent(llm)
    experiment_designer = ExperimentDesignerAgent(llm)
    synthesizer = create_synthesizer_node(llm)
    # create graph
    workflow = StateGraph(AgentState)
    # add nodes
    workflow.add_node("router", router)
    workflow.add_node("memory_retrieval", memory_retrieval_node)
    workflow.add_node("research_analyst", research_analyst)
    workflow.add_node("hypothesis_generator", hypothesis_generator)
    workflow.add_node("experiment_designer", experiment_designer)
    workflow.add_node("synthesizer", synthesizer)
    workflow.add_node("memory_update", memory_update_node)
    # set entry point
    workflow.set_entry_point("router")
    # add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "memory_retrieval": "memory_retrieval",
            "research_analyst": "research_analyst",
            "hypothesis_generator": "hypothesis_generator",
            "experiment_designer": "experiment_designer"
        }
    )
    # add conditional edges from memory
    workflow.add_conditional_edges(
        "memory_retrieval",
        route_after_memory,
        {
            "research_analyst": "research_analyst",
            "hypothesis_generator": "hypothesis_generator",
            "experiment_designer": "experiment_designer"
        }
    )
    # add conditional edges from research analyst
    workflow.add_conditional_edges(
        "research_analyst",
        route_after_research,
        {
            "hypothesis_generator": "hypothesis_generator",
            "synthesizer": "synthesizer"
        }
    )
    # add conditional edges from hypothesis generator
    workflow.add_conditional_edges(
        "hypothesis_generator",
        route_after_hypothesis,
        {
            "experiment_designer": "experiment_designer",
            "synthesizer": "synthesizer"
        }
    )
    # experiment designer always goes to synthesizer
    workflow.add_edge("experiment_designer", "synthesizer")
    # synthesizer goes to memory update
    workflow.add_edge("synthesizer", "memory_update")
    # memory update ends the graph
    workflow.add_edge("memory_update", END)
    # compile graph
    return workflow.compile()


def run_query(graph, query: str, verbose: bool = True) -> dict:
    # run a query through the multi-agent graph
    initial_state = create_initial_state(query)
    if verbose:
        print(f"\n{'---'}")
        print(f"QUERY: {query}")
        print(f"{'---'}\n")
    # run the graph
    final_state = graph.invoke(initial_state)
    if verbose:
        print(f"\n{'---'}")
        print("FINAL RESPONSE:")
        print(f"{'---'}")
        print(final_state.get("final_response", "No response generated"))
        print(f"\n{'---'}")
        print(f"Agents activated: {', '.join(set(final_state.get('agents_activated', [])))}")
        print(f"{'---'}\n")
    return final_state
