import json


# ACTION_PROMPT: 建模对手 → 评估风险 → 枚举候选 → 表态意图 → 提交动作
ACTION_PROMPT = """你是一个德州扑克玩家。每次决策都要以严格 JSON 输出 5 个字段：
{
  "opponent_models": {对手id: 一句话当下判断},
  "risk": {"worst": "...", "best": "...", "likely": "..."},
  "candidates": [{"action": {...}, "reason": "..."}, ...],
  "intent": "你想让对手对你产生什么印象",
  "action": {"type": "fold"|"check"|"call"|"raise", "amount": int (仅 raise 用)}
}

action.type 必须从 legal_actions 中选一个；raise 的 amount 是"加注到的总额"，
必须落在 legal_actions 给出的 [min_amount, max_amount] 区间。只输出 JSON。"""


REFLECTION_PROMPT = """你刚打完德州扑克的一手。请你基于这手的经过更新你的认知。

复盘原则：
1. 目标是"提炼可迁移到未来类似情形的判断"，不是叙述本手发生了什么。
2. 本手输赢由"决策质量 × 随机牌运"共同决定——请分别评估二者，
   不要因为赢了就肯定决策、因为输了就否定决策。
3. 当本手没有提供新信息时，"不更新"是合法且常常正确的输出。

输出严格 JSON，字段由具体提问决定。只输出 JSON。"""


def _outcome(final_state, my_id):
    for w in final_state.get("winners", []) or []:
        if w["player_id"] == my_id:
            return "win"
    return "loss"


def _won_amount(final_state, my_id):
    total = 0
    for w in final_state.get("winners", []) or []:
        if w["player_id"] == my_id:
            total += int(w.get("amount", 0) or 0)
    return total


def _fmt_action(act):
    if not isinstance(act, dict):
        return str(act)
    t = act.get("type", "?")
    if t == "raise" and act.get("amount") is not None:
        return f"{t} {act['amount']}"
    if t == "call" and act.get("amount") is not None:
        return f"{t} {act['amount']}"
    return t


def build_base_prompts(state):
    return state, ACTION_PROMPT, REFLECTION_PROMPT


def build_user_prompt(state, extra_sections=None):
    """ user message """
    me_id = state["current_player_id"]
    me = next(p for p in state["players"] if p["id"] == me_id)
    others = [p for p in state["players"] if p["id"] != me_id]

    lines = [
        f"第 {state['hand_index']} 手 · {state['phase']} 阶段",
        f"底池: {state['pot']}  本街最高下注: {state['current_bet_this_round']}  公共牌: {state['community_cards']}",
        "",
        f"你: id={me['id']}, hole={me['hole_cards']}, stack={me['stack']}, "
        f"current_bet={me['current_bet']}, total_committed={me['total_committed']}",
        "对手：",
    ]
    for p in others:
        flag = " [已弃牌]" if p["folded"] else (" [出局]" if p["busted"] else "")
        lines.append(
            f"  {p['id']}: stack={p['stack']}, current_bet={p['current_bet']}, "
            f"total_committed={p['total_committed']}{flag}"
        )

    if state["action_history"]:
        lines.append("")
        lines.append("本手已发生：")
        for ev in state["action_history"]:
            lines.append(f"  [{ev['phase']}] {ev['player_id']} -> {ev['action']}")

    lines.append("")
    lines.append(f"合法动作: {state['legal_actions']}")

    for title, body in (extra_sections or {}).items():
        if not body:
            continue
        lines.append("")
        lines.append(f"=== {title} ===")
        lines.append(body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, indent=2))

    lines.append("")
    lines.append("请输出 5 字段 JSON。")
    return "\n".join(lines)


def summarize_hand(working_buffer, final_state, my_id):
    """
    (a) 全部玩家的完整动作序列；
    (b) 我的 intent；
    (c) 终局结果与对手揭示牌。
    """
    hand_idx = final_state["hand_index"]
    lines = [f"=== hand {hand_idx} ==="]
    lines.append(f"公共牌: {final_state['community_cards']}")
    lines.append("")
    lines.append("完整动作序列:")
    for ev in final_state.get("action_history", []):
        suffix = " (me)" if ev["player_id"] == my_id else ""
        lines.append(f"  [{ev['phase']}] {ev['player_id']}{suffix} {_fmt_action(ev['action'])}")

    lines.append("")
    lines.append("我的内心活动:")
    for turn in working_buffer:
        parsed = turn.get("parsed") or {}
        lines.append(
            f"  [{turn['phase']}] intent={parsed.get('intent')!r}  action={_fmt_action(turn['action'])}"
        )

    me = next((p for p in final_state["players"] if p["id"] == my_id), None)
    committed = me["total_committed"] if me else 0
    stack = me["stack"] if me else 0
    won = _won_amount(final_state, my_id)
    note_winners = "; ".join(
        f"{w['player_id']}={w.get('amount', 0)}" for w in (final_state.get("winners") or [])
        if w["player_id"] != my_id
    )
    if won > 0:
        lines.append("")
        lines.append(f"结局: stack={stack}, 本手投入 {committed}, 赢得 {won}")
    else:
        extra = f" (赢家: {note_winners})" if note_winners else ""
        lines.append("")
        lines.append(f"结局: stack={stack}, 本手投入 {committed}, 未赢{extra}")

    revealed = [
        f"  {p['id']}: hole={p['hole_cards']}"
        for p in final_state["players"]
        if p["id"] != my_id and not p["folded"] and p["hole_cards"]
    ]
    if revealed:
        lines.append("对手揭示:")
        lines.extend(revealed)
    else:
        lines.append("对手揭示: 无")

    return "\n".join(lines)
