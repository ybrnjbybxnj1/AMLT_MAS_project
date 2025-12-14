import re
import json
import numpy as np
from typing import List
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import Hypothesis, Paper

TRIZ_PRINCIPLES = [
    "Segmentation", "Taking out", "Local quality", "Asymmetry",
    "Merging", "Universality", "Nested doll", "Anti-weight",
    "Preliminary anti-action", "Preliminary action", "Beforehand cushioning",
    "Equipotentiality", "The other way round", "Spheroidality",
    "Dynamics", "Partial or excessive actions", "Another dimension",
    "Mechanical vibration", "Periodic action", "Continuity of useful action",
    "Skipping", "Blessing in disguise", "Feedback", "Intermediary",
    "Self-service", "Copying", "Cheap short-living objects", "Mechanics substitution",
    "Pneumatics and hydraulics", "Flexible shells and thin films",
    "Porous materials", "Color changes", "Homogeneity", "Discarding and recovering",
    "Parameter changes", "Phase transitions", "Thermal expansion",
    "Strong oxidants", "Inert atmosphere", "Composite materials"
]

def generate_hypothesis_tool(focus: str, trends: List[str], gaps: List[str], llm) -> Hypothesis:
    # generate a research hypothesis using TRIZ methodology
    print(f"[TOOL] Generating hypothesis for: {focus}...")
    # select relevant TRIZ principles
    relevant_principles = TRIZ_PRINCIPLES[:10]  # use first 10 for simplicity
    prompt = f"""
    Generate a novel research hypothesis using TRIZ methodology.

    Research Focus: {focus}

    Current Trends:
    {chr(10).join(f'- {t}' for t in trends[:3])}

    Research Gaps/Opportunities:
    {chr(10).join(f'- {g}' for g in gaps[:3])}

    Available TRIZ Principles:
    {chr(10).join(f'- {p}' for p in relevant_principles)}

    Generate a hypothesis that:
    1. Addresses at least one gap
    2. Builds on current trends
    3. Applies 1-3 TRIZ principles creatively
    4. Is specific and testable

    Respond with JSON:
    {{
        "statement": "Your hypothesis statement (specific, testable, 20+ chars)",
        "triz_principles": ["principle1", "principle2"],
        "rationale": "Why this matters and how it addresses the gap (20+ chars)",
        "novelty_score": 7
    }}
    """
    try:
        response = llm.invoke([
            SystemMessage(content="You are a creative research scientist using TRIZ. Respond only with valid JSON."),
            HumanMessage(content=prompt)
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        data = json.loads(content)
        return Hypothesis(**data)
    except Exception as e:
        print(f"[TOOL] Hypothesis generation fallback: {e}")
        return Hypothesis(
            statement=f"Applying {relevant_principles[0]} to {focus} will improve research outcomes by addressing identified gaps",
            triz_principles=[relevant_principles[0], relevant_principles[1]],
            rationale=f"This hypothesis addresses the gap in {gaps[0] if gaps else 'current approaches'} by leveraging emerging trends",
            novelty_score=6
        )


def calculate_novelty(hypothesis: str, papers: List[Paper]) -> dict:
    # calculate novelty score based on keyword overlap with existing papers
    print(f"[TOOL] calculating novelty score...")
    if not papers:
        return {
            "score": 7,
            "method": "default",
            "reason": "no papers to compare - assuming moderate novelty"
        } 
    # extract words from hypothesis
    hyp_words = set(re.findall(r'\b\w{4,}\b', hypothesis.lower()))
    # calculate overlap with each paper
    overlaps = []
    for paper in papers[:10]:
        paper_text = f"{paper.title} {paper.abstract}".lower()
        paper_words = set(re.findall(r'\b\w{4,}\b', paper_text))
        if hyp_words and paper_words:
            overlap = len(hyp_words & paper_words) / len(hyp_words)
            overlaps.append(overlap)
    if not overlaps:
        return {
            "score": 7,
            "method": "default",
            "reason": "could not calculate overlap"
        }
    avg_overlap = np.mean(overlaps)
    novelty = int((1 - avg_overlap) * 10)
    novelty = max(1, min(10, novelty))
    return {
        "score": novelty,
        "method": "keyword_overlap",
        "avg_overlap": round(avg_overlap, 2),
        "papers_compared": len(overlaps),
        "reason": f"average keyword overlap with {len(overlaps)} papers: {round(avg_overlap*100, 1)}%"
    }
