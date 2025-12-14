import requests
import xml.etree.ElementTree as ET
import re
import time
import unicodedata
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.tools import tool
from ..state import Paper, TrendAnalysis, ContradictionAnalysis


def clean_text(text):
    # clean text by normalizing unicode and removing extra whitespace
    if not text: return ""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return ' '.join(text.split())

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.RequestException, requests.Timeout)),
    reraise=True
)
def search_arxiv(query: str, max_results: int = 10) -> dict:
    # search arXiv for papers matching the query
    print(f"[TOOL] ArXiv search: '{query}...'")
    base_url = "http://export.arxiv.org/api/query"
    params = {
        'search_query': f'all:{query}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
            id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
            
            if title_elem is not None and summary_elem is not None:
                clean_title = clean_text(title_elem.text)
                clean_abstract = clean_text(summary_elem.text)
                year = None
                if published_elem is not None:
                    year_match = re.search(r'(\d{4})', published_elem.text)
                    if year_match:
                        year = int(year_match.group(1))
                papers.append(Paper(
                    title=clean_title,
                    abstract=clean_abstract,
                    year=year,
                    source="arxiv",
                    url=id_elem.text if id_elem is not None else None
                ))
        method_keywords = [
            'reinforcement learning', 'deep learning', 'neural network',
            'machine learning', 'transformer', 'attention', 'optimization',
            'bayesian', 'graph neural', 'generative', 'diffusion'
        ]
        abstracts_text = ' '.join([p.abstract.lower() for p in papers])
        methods = [kw.title() for kw in method_keywords if kw in abstracts_text][:3]
        print(f"[TOOL] found {len(papers)} papers")
        return {
            "papers_found": len(papers),
            "key_topics": [p.title for p in papers[:5]],
            "recent_methods": methods if methods else ["Novel approach"],
            "papers": papers
        }
    except requests.RequestException as e:
        print(f"[TOOL] ArXiv API error: {e}")
        return {
            "papers_found": 0,
            "key_topics": [],
            "recent_methods": [],
            "papers": [],
            "error": str(e)
        }


@tool
def analyze_trends_tool(literature_data: dict, focus: str, llm) -> TrendAnalysis:
    # analyze research trends from literature data using LLM
    from langchain_core.messages import SystemMessage, HumanMessage
    import json 
    print(f"[TOOL] analyzing trends for: {focus[:50]}...")
    papers = literature_data.get('papers', [])
    if papers:
        papers_context = "\n".join([
            f"- {p.title} ({p.year or 'N/A'}): {p.abstract[:150]}..."
            for p in papers[:5]
        ])
    else:
        papers_context = '\n'.join(f"- {t}" for t in literature_data.get('key_topics', [])) 
    prompt = f"""
    Analyze research trends from these papers/topics:

    {papers_context}

    Research Focus: {focus}

    Identify:
    1. 3-5 current research trends
    2. 2-3 emerging directions
    3. Your confidence level (high/medium/low)

    Respond with JSON:
    {{"trends": ["trend1", "trend2", ...], "emerging_directions": ["dir1", "dir2"], "confidence": "high|medium|low"}}
    """
    try:
        response = llm.invoke([
            SystemMessage(content="You are a research trend analyst. Respond only with valid JSON."),
            HumanMessage(content=prompt)
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        data = json.loads(content)
        return TrendAnalysis(**data)
    except Exception as e:
        print(f"[TOOL] trend analysis fallback: {e}")
        return TrendAnalysis(
            trends=literature_data.get('key_topics', ["Emerging research"])[:3],
            emerging_directions=["Novel applications", "Cross-domain integration"],
            confidence="medium"
        )


@tool  
def find_gaps_tool(literature_data: dict, focus: str, llm) -> ContradictionAnalysis:
    # identify research gaps and contradictions from literature
    from langchain_core.messages import SystemMessage, HumanMessage
    import json
    print(f"[TOOL] finding research gaps for: {focus[:50]}...")
    papers = literature_data.get('papers', [])
    if papers:
        papers_context = "\n".join([
            f"- {p.title}: {p.abstract}..."
            for p in papers[:5]
        ])
    else:
        papers_context = '\n'.join(f"- {t}" for t in literature_data.get('key_topics', []))
    prompt = f"""Analyze these papers/topics for research gaps:

    {papers_context}

    Research Focus: {focus}

    Identify:
    1. 2-3 contradictions or limitations in current approaches
    2. 2-3 unsolved problems
    3. 2-3 research opportunities

    Respond with JSON:
    {{"contradictions": ["c1", "c2"], "unsolved_problems": ["p1", "p2"], "opportunities": ["o1", "o2"]}}
    """
    try:
        response = llm.invoke([
            SystemMessage(content="You are a critical research analyst. Respond only with valid JSON."),
            HumanMessage(content=prompt)
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        data = json.loads(content)
        return ContradictionAnalysis(**data)
    except Exception as e:
        print(f"[TOOL] gap analysis fallback: {e}")
        return ContradictionAnalysis(
            contradictions=["Limited scalability", "Lack of real-world validation"],
            unsolved_problems=["Generalization across domains", "Computational efficiency"],
            opportunities=["Novel hybrid approaches", "Cross-disciplinary methods"]
        )
