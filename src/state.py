from typing import TypedDict, List, Optional, Annotated, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import operator

class Paper(BaseModel):
    # individual paper data
    title: str
    abstract: str = ""
    year: Optional[int] = None
    source: str = "arxiv"
    url: Optional[str] = None


class QueryClassification(BaseModel):
    # router agent's classification output
    query_type: Literal["conceptual", "design", "implementation", "planning"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str
    needs_memory: bool
    is_followup: bool
    target_agents: List[str]


class TrendAnalysis(BaseModel):
    # trend analysis output
    trends: List[str]
    emerging_directions: List[str]
    confidence: Literal["high", "medium", "low"]


class ContradictionAnalysis(BaseModel):
    # research gaps analysis
    contradictions: List[str]
    unsolved_problems: List[str]
    opportunities: List[str]


class Hypothesis(BaseModel):
    # generated hypothesis
    statement: str
    triz_principles: List[str]
    rationale: str
    novelty_score: int


class ExperimentPlan(BaseModel):
    # experiment design output
    feasibility: Literal["high", "medium", "low"]
    steps: List[str]
    resources: List[str]
    duration: str
    challenges: List[str]


class MemoryEntry(BaseModel):
    # single memory entry
    query: str
    response_summary: str
    agents_used: List[str]
    key_findings: List[str] = []


class UserProfile(BaseModel):
    # user research profile
    research_interests: List[str] = []
    expertise_level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    previous_hypotheses: List[str] = []


def merge_lists(left: Optional[List], right: Optional[List]) -> List:
    # merge two lists, handling None values
    left = left or []
    right = right or []
    return left + right


def merge_optional(left: Optional[any], right: Optional[any]) -> Optional[any]:
    # return first non-None value
    return right if right is not None else left


class AgentState(TypedDict):
    # shared state that flows through the multi-agent graph
    # input
    user_query: str
    # router output
    classification: Optional[QueryClassification]
    current_agent: Optional[str]
    agents_activated: Annotated[List[str], operator.add]
    # research analyst output
    literature_data: Optional[dict]
    papers: Annotated[List[Paper], merge_lists]
    trends: Optional[TrendAnalysis]
    gaps: Optional[ContradictionAnalysis]
    # hypothesis generator output
    hypothesis: Optional[Hypothesis]
    novelty_score: Optional[dict]
    # experiment designer output
    experiment_plan: Optional[ExperimentPlan]
    feasibility_score: Optional[dict]
    # final output
    final_response: Optional[str]
    messages: Annotated[List[str], operator.add]
    # memory management
    session_history: Annotated[List[MemoryEntry], merge_lists]
    user_profile: Optional[UserProfile]
    notes: Annotated[List[str], operator.add]
    memory_context: Optional[str]


def create_initial_state(query: str) -> AgentState:
    # create initial state for a new query
    return AgentState(
        user_query=query,
        classification=None,
        current_agent=None,
        agents_activated=[],
        literature_data=None,
        papers=[],
        trends=None,
        gaps=None,
        hypothesis=None,
        novelty_score=None,
        experiment_plan=None,
        feasibility_score=None,
        final_response=None,
        messages=[],
        session_history=[],
        user_profile=None,
        notes=[],
        memory_context=None
    )
