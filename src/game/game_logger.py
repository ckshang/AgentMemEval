"""
牌局级日志：每手结束写一行到 hands.csv，每个动作一行到 actions.csv。
agent 自己的 LLM 思考 / 记忆落盘归各 agent 自己管，这里只管 controller 视角。
"""
import csv

from pathlib import Path


_ACTION_HEADER = [
    "hand_index", "phase", "player_id",
    "action_type", "action_amount",
    "pot_before",
]
_HAND_HEADER = [
    "hand_index", "phase_final", "pot_final",
    "num_winners", "winner_ids", "winner_amounts", "winner_hand_ranks",
    "went_to_showdown", "num_actions", "showdown_ranks",
]
_STACKS_HEADER = ["hand_index", "player_id", "stack"]


def write_hand_rows(out_dir, controller, final_state):
    """每手结束调一次，把 action_history、本手结果、每玩家末态 stack 落盘。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actions_path = out_dir / "actions.csv"
    hands_path = out_dir / "hands.csv"
    stacks_path = out_dir / "stacks.csv"

    # --- actions.csv：每个动作一行 ---
    new_actions = not actions_path.exists()
    with open(actions_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_actions:
            w.writerow(_ACTION_HEADER)
        for ev in final_state["action_history"]:
            act = ev["action"]
            w.writerow([
                final_state["hand_index"],
                ev["phase"],
                ev["player_id"],
                act["type"],
                act.get("amount", ""),
                "",   # pot_before 暂不计算（要的话需 controller 同步记录）
            ])

    # --- hands.csv：这手一行 ---
    new_hands = not hands_path.exists()
    winners = final_state.get("winners") or []
    went_to_showdown = any(w.get("hand_rank") for w in winners)
    sd_ranks = final_state.get("showdown_ranks") or {}
    sd_ranks_str = ";".join(f"{pid}={rk}" for pid, rk in sd_ranks.items())
    with open(hands_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_hands:
            w.writerow(_HAND_HEADER)
        w.writerow([
            final_state["hand_index"],
            final_state["phase"],
            sum(x.get("amount", 0) for x in winners),
            len(winners),
            ";".join(x["player_id"] for x in winners),
            ";".join(str(x.get("amount", "")) for x in winners),
            ";".join(str(x.get("hand_rank") or "") for x in winners),
            went_to_showdown,
            len(final_state["action_history"]),
            sd_ranks_str,
        ])

    # --- stacks.csv：每个玩家本手末 stack 一行（长格式，便于 pandas 透视画图）---
    new_stacks = not stacks_path.exists()
    with open(stacks_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_stacks:
            w.writerow(_STACKS_HEADER)
        for p in final_state["players"]:
            w.writerow([final_state["hand_index"], p["id"], p["stack"]])
