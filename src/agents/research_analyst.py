import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from ..state import AgentState, TrendAnalysis, ContradictionAnalysis
from ..tools.literature_tools import search_arxiv
from ..llm_utils import invoke_with_parser


ANALYST_SYSTEM_PROMPT = """
You are a Research Analyst Agent specializing in literature review and trend analysis.

Your responsibilities:
1. Search and analyze academic literature (via arXiv)
2. Identify current research trends
3. Find research gaps and opportunities
4. Summarize key findings for other agents

You have access to:
- ArXiv search tool
- Trend analysis capabilities
- Gap/contradiction identification

Provide thorough, evidence-based analysis.
"""


def research_analyst_node(state: AgentState, llm) -> dict:
    # research analyst agent node that performs literature analysis
    query = state["user_query"]
    classification = state.get("classification")
    memory_context = state.get("memory_context", "")
    print(f"[RESEARCH ANALYST] analyzing: {query}...")
    # extract keywords for search
    keyword_prompt = f"""Extract 3-5 search keywords from this research query:

    Query: {query}

    {f'Context from memory: {memory_context}' if memory_context else ''}

    Respond with JSON: {{"keywords": ["kw1", "kw2", "kw3"]}}"""
    try:
        kw_response = llm.invoke([
            SystemMessage(content="Extract research keywords. Respond only with JSON."),
            HumanMessage(content=keyword_prompt)
        ])
        kw_content = kw_response.content.strip()
        if kw_content.startswith("```"):
            kw_content = re.sub(r'^```\w*\n?', '', kw_content)
            kw_content = re.sub(r'\n?```$', '', kw_content)
        keywords = json.loads(kw_content).get("keywords", [query.split()[:3]])
    except:
        keywords = query.split()[:5]
    # search arXiv
    search_query = ' '.join(keywords[:3])
    literature_data = search_arxiv(search_query, max_results=10)
    papers = literature_data.get("papers", [])
    # analyze trends
    trends = _analyze_trends(literature_data, query, llm)
    # find gaps
    gaps = _find_gaps(literature_data, query, llm)
    # generate summary message
    summary = f"research analyst: found {literature_data['papers_found']} papers. "
    summary += f"identified {len(trends.trends)} trends and {len(gaps.opportunities)} opportunities."
    return {
        "literature_data": literature_data,
        "papers": papers,
        "trends": trends,
        "gaps": gaps,
        "current_agent": "research_analyst",
        "agents_activated": ["research_analyst"],
        "messages": [summary],
        "notes": [f"Key trends: {', '.join(trends.trends[:2])}"]
    }


def _analyze_trends(literature_data: dict, focus: str, llm) -> TrendAnalysis:
    # analyze trends from literature data 
    papers = literature_data.get('papers', [])
    trend_parser = PydanticOutputParser(pydantic_object=TrendAnalysis)
    if papers:
        pc = "\n".join([f"- {p.title}: {p.abstract}..." for p in papers])
    else:
        pc = "No papers found"
    trend_prompt = f"""
    Analyze research trends from these papers.

    Papers:
    {pc}

    Research focus: {focus}

    Identify current trends, emerging directions, and your confidence level."""
    try:
        trends = invoke_with_parser(llm, trend_parser, trend_prompt)
        return trends
    except Exception as e:
        print(f"[RESEARCH] trend parse failed: {e}")
        return TrendAnalysis(
            trends=["Emerging AI research"],
            emerging_directions=["Novel methodologies"],
            confidence="medium"
        )


def _find_gaps(literature_data: dict, focus: str, llm) -> ContradictionAnalysis:
    # find research gaps from literature
    papers = literature_data.get('papers', [])
    gap_parser = PydanticOutputParser(pydantic_object=ContradictionAnalysis)
    if papers:
        pc = "\n".join([f"- {p.title}: {p.abstract}..." for p in papers])
    else:
        pc = "No papers found"
    gap_prompt = f"""
    Find research gaps and opportunities from these papers.

    Papers:
    {pc}

    Identify contradictions in the literature, unsolved problems, and research opportunities.
    """
    try:
        gaps = invoke_with_parser(llm, gap_parser, gap_prompt)
        return gaps
    except Exception as e:
        print(f"[RESEARCH] gap parse failed: {e}")
        return ContradictionAnalysis(
            contradictions=["Limited scope"],
            unsolved_problems=["Scalability"],
            opportunities=["Novel approaches"]
        )


class ResearchAnalystAgent:
    # research analyst agent class wrapper
    def __init__(self, llm):
        self.llm = llm
        self.name = "research_analyst"
    def __call__(self, state: AgentState) -> dict:
        return research_analyst_node(state, self.llm)
