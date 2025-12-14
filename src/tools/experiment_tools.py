import re
import json
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import ExperimentPlan


def design_experiment_tool(hypothesis: str, resources_available: List[str], llm) -> ExperimentPlan:
    # designs an experiment to test a hypothesis
    print(f"[TOOL] designing experiment for hypothesis...")
    prompt = f"""
    Design an experiment to test this hypothesis:

    Hypothesis: {hypothesis}

    Available resources/constraints:
    {chr(10).join(f'- {r}' for r in resources_available) if resources_available else '- Standard research computing resources'}

    Create a practical experiment plan with:
    1. 3-7 exact steps
    2. Required resources
    3. Expected duration
    4. Potential challenges
    5. Feasibility assessment (high/medium/low)

    Respond with JSON:
    {{
        "feasibility": "high|medium|low",
        "steps": ["step1", "step2", "step3"],
        "resources": ["resource1", "resource2"],
        "duration": "X weeks/months",
        "challenges": ["challenge1", "challenge2"]
    }}"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an experimental research designer. Respond only with valid JSON."),
            HumanMessage(content=prompt)
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        data = json.loads(content)
        return ExperimentPlan(**data)
    except Exception as e:
        print(f"[TOOL] experiment design fallback: {e}")
        return ExperimentPlan(
            feasibility="medium",
            steps=[
                "Define experimental setup and variables",
                "Prepare data collection methodology",
                "Implement baseline and proposed approach",
                "Run experiments and collect results",
                "Analyze results and draw conclusions"
            ],
            resources=["Computing resources", "Relevant datasets", "Analysis tools"],
            duration="4-8 weeks",
            challenges=["Data availability", "Computational constraints", "Result interpretation"]
        )


def calculate_feasibility(experiment: ExperimentPlan) -> dict:
    # calculate feasibility score based on experiment characteristics
    print(f"[TOOL] calculating feasibility score...")
    score = 10
    details = []
    # Check for expensive resources
    expensive_resources = [
        'mri', 'fmri', 'supercomputer', 'quantum computer',
        'gene sequencing', 'particle accelerator', 'satellite',
        'a100', 'h100', 'cluster'
    ]
    resources_text = ' '.join(experiment.resources).lower()
    expensive_count = sum(1 for r in expensive_resources if r in resources_text)
    if expensive_count > 0:
        score -= expensive_count * 2
        details.append(f"Expensive resources: -{expensive_count * 2}")
    # check duration
    duration_lower = experiment.duration.lower()
    if 'year' in duration_lower:
        years_match = re.search(r'(\d+)', duration_lower)
        if years_match:
            years = int(years_match.group(1))
            if years > 2:
                score -= 3
                details.append(f"Long duration ({years} years): -3")
            elif years > 1:
                score -= 1
                details.append(f"Moderate duration ({years} years): -1")
    elif 'month' in duration_lower:
        months_match = re.search(r'(\d+)', duration_lower)
        if months_match:
            months = int(months_match.group(1))
            if months > 6:
                score -= 2
                details.append(f"Extended duration ({months} months): -2")
    # check complexity
    if len(experiment.steps) > 7:
        score -= 1
        details.append(f"Many steps ({len(experiment.steps)}): -1")
    if len(experiment.challenges) > 4:
        score -= 1
        details.append(f"Many challenges ({len(experiment.challenges)}): -1")
    score = max(1, min(10, score))
    if score >= 7:
        category = "high"
    elif score >= 4:
        category = "medium"
    else:
        category = "low"
    return {
        "category": category,
        "score": score,
        "details": details,
        "reason": f"Feasibility score: {score}/10 based on resources, duration, and complexity"
    }


def estimate_duration(steps: List[str]) -> dict:
    # estimate experiment duration using PERT-like analysis
    print(f"[TOOL] estimating duration...")
    # task patterns with (optimistic, likely, pessimistic) weeks
    task_patterns = {
        'recruit|participant|subject': (2, 4, 8),
        'setup|install|configure|calibrate': (1, 2, 4),
        'collect|gather|measure|record|observe': (4, 8, 16),
        'analyze|process|evaluate|assess': (2, 4, 8),
        'write|document|report|publish': (2, 3, 6),
        'train|learn|practice|fine-tune': (1, 2, 4),
        'test|trial|experiment|run': (3, 6, 12),
        'develop|create|build|design|implement': (2, 4, 8)
    }
    total_weeks = 0
    breakdown = []
    for step in steps:
        step_lower = step.lower()
        matched = False
        for pattern, (opt, likely, pess) in task_patterns.items():
            if re.search(pattern, step_lower):
                expected = (opt + 4*likely + pess) / 6
                total_weeks += expected
                breakdown.append({"step": step[:50], "weeks": round(expected, 1)})
                matched = True
                break
        if not matched:
            total_weeks += 2
            breakdown.append({"step": step[:50], "weeks": 2.0})
    # add 20% buffer
    total_with_buffer = total_weeks * 1.2
    if total_with_buffer > 12:
        duration_str = f"{int(total_with_buffer / 4)}-{int(total_with_buffer / 4 * 1.3)} months"
    else:
        duration_str = f"{int(total_weeks)}-{int(total_with_buffer)} weeks"
    return {
        "duration": duration_str,
        "base_weeks": round(total_weeks, 1),
        "with_buffer_weeks": round(total_with_buffer, 1),
        "breakdown": breakdown,
        "method": "PERT analysis with 20% buffer"
    }
