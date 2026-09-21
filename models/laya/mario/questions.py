"""Canonical Laya question schema for Mario control -- single source of truth for both
inference (eval.py, via RLAgent.system_one) and training (finetune.py, via rl_common.encode_record).
"""

QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": "Mario is playing a 2D side-scrolling platformer. Given the current game "
                        "state, which horizontal direction should Mario move to make progress and "
                        "avoid danger?",
        "criteria": {
            "left": "move left, away from the goal",
            "right": "move right, towards the goal",
            "neutral": "stand still, do not move horizontally",
        },
    },
    "jump": {
        "type": "noul",
        "instructions": "Given the current game state, should Mario jump right now (to clear a gap, "
                        "an obstacle, or an enemy)?",
    },
}

MOVE_LABELS = list(QUESTIONS["move"]["criteria"].keys())  # ["left", "right", "neutral"]
