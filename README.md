# Multi-agent research hypothesis planner

A LangGraph-based multi-agent system that assists researchers with literature analysis, hypothesis generation using TRIZ methodology, and experiment design.

## Idea

This project implements a multi-agent orchestration system where specialized AI agents collaborate to help researchers in following areas:
- Analyze academic literature from arXiv;
- Identify research trends and gaps;
- Generate novel hypotheses using TRIZ (Theory of Inventive Problem Solving) principles;
- Design practical experiments with feasibility assessments.

The system uses LangGraph for agent coordination and memory to maintain context across conversations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Router Agent                              │
│  Classifies query type: conceptual/design/implementation/planning│
│  Determines if memory retrieval is needed                        │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ Memory Retrieval  │ │ Research Analyst  │ │Experiment Designer│
│ (if follow-up)    │ │ ArXiv search      │ │ Experiment plans  │
└───────────────────┘ │ Trend analysis    │ │ Feasibility calc  │
            │         │ Gap identification│ └───────────────────┘
            │         └───────────────────┘           │
            │                   │                     │
            │                   ▼                     │
            │         ┌───────────────────┐           │
            │         │Hypothesis Generator│          │
            │         │ TRIZ methodology  │           │
            │         │ Novelty scoring   │           │
            │         └───────────────────┘           │
            │                   │                     │
            └───────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Synthesizer                               │
│           Combines all agent outputs into final response         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Memory Update                              │
│              Saves interaction to memory.json                    │
└─────────────────────────────────────────────────────────────────┘
```

## Agents Description

| Agent | Role | Tools/Capabilities |
|-------|------|-------------------|
| Router | Classifies queries and routes to appropriate specialists | Query classification, memory context check |
| Research Analyst | Literature review and trend analysis | ArXiv API search, trend analysis, gap identification |
| Hypothesis Generator | Creates novel research hypotheses | TRIZ principles (40 inventive principles), novelty scoring |
| Experiment Designer | Designs practical experiments | Feasibility calculation, duration estimation (PERT analysis) |
| Memory Manager | Maintains conversation context | JSON-based persistent storage, context retrieval |
| Synthesizer | Combines all outputs into coherent response | LLM-based synthesis |

## Design Scheme

The system follows a conditional graph execution pattern:

1. Entry: all queries enter through the Router;
2. Conditional routing: based on query type:
   - conceptual -> Research analyst -> Synthesizer
   - design -> Research analyst -> Hypothesis generator -> Synthesizer
   - implementation -> Experiment designer -> Synthesizer
   - planning -> Research analyst -> Hypothesis generator -> Experiment designer -> Synthesizer;
3. Memory integration: follow-up queries trigger memory retrieval before routing;
4. Exit: all paths end with Memory Update -> END

## Project structure

```
AMLT_MAS_project/
├── src/
│   ├── agents/
│   │   ├── router.py # Query classification
│   │   ├── research_analyst.py # Literature analysis
│   │   ├── hypothesis_generator.py # TRIZ-based hypothesis
│   │   ├── experiment_designer.py # Experiment planning
│   │   └── memory_manager.py # Memory update
│   ├── tools/
│   │   ├── literature_tools.py # ArXiv search
│   │   ├── hypothesis_tools.py # TRIZ, novelty calc
│   │   └── experiment_tools.py # Feasibility, duration
│   ├── graph.py # LangGraph construction
│   ├── state.py # Pydantic models & state
│   ├── llm_utils.py # LLM retry utilities
│   └── main.py # CLI entry point
├── Lab2_MultiAgent_Hypothesis_Planner.ipynb # Interactive demo
├── memory.json # Memory storage
├── requirements.txt # Dependencies
└── .env # API configuration
```

## Installation

1. Clone the repository and navigate to the project folder

2. Create a virtual environment (recommended by best practices and me):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or: source .venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables in `.env`:
   ```env
   LITELLM_BASE_URL=your_llm_endpoint
   LITELLM_API_KEY=your_api_key
   MODEL_NAME=qwen3-32b
   ```

## Quick start demo

### Option 1: Jupyter notebook (Recommended)
```bash
jupyter notebook Lab2_MultiAgent_Hypothesis_Planner.ipynb
```
Run cells sequentially to see the multi-agent system in action.

### Option 2: Command Line
```bash
# Single query
python src/main.py "What are benefits of multi-agent systems for LLM orchestration?"

# Interactive mode
python src/main.py --interactive
```

### Option 3: Python script
```python
from src.graph import build_graph, run_query
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="...", api_key="...", model="qwen3-32b")
graph = build_graph(llm)
result = run_query(graph, "Design a hypothesis about agent communication")
```

## Results reflection

### What worked well
- Routing with fallbacks: when the Router hallucinated agent names, the normalization mapping and fallback mechanisms prevented crashes
- Memory for follow-ups: expanding on previous hypotheses successfully retrieved context (e.g., TRIZ principles from earlier queries)
- Real literature integration: ArXiv search provided actual academic papers for analysis

### Limitations & areas for improvement
- Agent name hallucination: LLM sometimes invented agent names requiring extensive hardcoded mapping
- Simple novelty/feasibility metrics: current keyword-overlap novelty and rule-based feasibility calculations are heuristics, not true intelligence
- No reviewer agent: a review step before synthesis could improve output quality
- JSON-based memory: vector storage would enable semantic retrieval of specific conversation segments
