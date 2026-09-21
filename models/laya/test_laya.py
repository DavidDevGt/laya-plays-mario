"""Smoke test for the locally downloaded Laya typed-decisions checkpoint."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("USE_TF", "0")

from rl_agent_api import RLAgent

MODEL_DIR = os.path.join(os.path.dirname(__file__), "typed-decisions")

agent = RLAgent(MODEL_DIR)
print(f"Loaded on device: {agent.device}")

state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}

result = agent.system_one(state, questions)
print(json.dumps(result, indent=2, ensure_ascii=False))
