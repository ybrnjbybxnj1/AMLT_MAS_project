import json
import os
from datetime import datetime
from typing import List, Optional
from ..state import AgentState, MemoryEntry, UserProfile

MEMORY_FILE = "memory.json"

class MemStore:
    # memory storage 
    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self.history: List[MemoryEntry] = self._load()
        self.notes: List[str] = []
    
    def _load(self) -> List[MemoryEntry]:
        # load memory from file if exists
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    return [MemoryEntry(**d) for d in data]
            except Exception as e:
                print(f"[MEMORY] error loading file: {e}")
        return []
    def _save(self):
        # save memory to file
        try:
            data = [entry.model_dump() for entry in self.history]
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[MEMORY] error saving file: {e}")
    def add(self, q: str, r: str, a: List[str], f: List[str] = None):
        # add a new memory entry
        entry = MemoryEntry(
            query=q,
            response_summary=r,
            agents_used=a,
            key_findings=f or []
        )
        self.history.append(entry)
        self._save()
    def context(self, q: str, n: int = 3) -> str:
        # get relevant context from memory for a query
        if not self.history:
            return ""
        return "\n".join([f"Q:{e.query[:60]}->R:{e.response_summary[:80]}" for e in self.history[-n:]])
# alias for backward compatibility
MemoryStore = MemStore
# global memory store instance
_memory_store: Optional[MemStore] = None
def get_memory_store() -> MemStore:
    # get or create memory store singleton
    global _memory_store
    if _memory_store is None:
        _memory_store = MemStore()
    return _memory_store
def memory_retrieval_node(state: AgentState) -> dict:
    # memory retrieval node which gets relevant context from memory
    query = state["user_query"]
    print(f"[MEMORY] retrieving context...")
    store = get_memory_store()
    ctx = store.context(query)
    print(f"[MEMORY] retrieved context: {len(ctx)} chars")
    return {
        "memory_context": ctx or None,
        "agents_activated": ["memory_manager"]
    }
def memory_update_node(state: AgentState) -> dict:
    # memory update node which saves interaction to memory
    store = get_memory_store()
    # collect key findings from all available state data (fixed logic from notebook)
    findings = []
    h = state.get("hypothesis")
    if h and h.statement:
        findings.append(f"Hypothesis: {h.statement[:80]}")
    tr = state.get("trends")
    if tr and tr.trends:
        findings.append(f"Trends: {', '.join(tr.trends[:3])}")
    ga = state.get("gaps")
    if ga and ga.opportunities:
        findings.append(f"Opportunities: {', '.join(ga.opportunities[:3])}")
    e = state.get("experiment_plan")
    if e and e.steps:
        findings.append(f"Experiment: {len(e.steps)} steps, {e.feasibility} feasibility")
    store.add(
        state["user_query"],
        state.get("final_response", "")[:150],
        list(set(state.get("agents_activated", []))),
        findings
    )
    print(f"[MEMORY] saved interaction with {len(findings)} key findings")
    return {"agents_activated": ["memory_manager"]}

class MemoryManagerAgent:
    # memory manager agent class wrapper
    def __init__(self):
        self.name = "memory_manager"
        self.store = get_memory_store()
    def retrieve(self, state: AgentState) -> dict:
        return memory_retrieval_node(state)
    def update(self, state: AgentState) -> dict:
        return memory_update_node(state)
