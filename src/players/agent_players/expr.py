"""
ExprAgent，经验型 memory。
"""
from pathlib import Path

from src.players.base_player import BasePlayer
from src.players.prompts import build_base_prompts, build_user_prompt, summarize_hand
from src.players.prompts import _outcome, _won_amount
from src.players.llms import call_llm, parse_response
from src.utils.file_storage import JSONLStore, _alive_opponents_snapshot, safe_json


INITIAL_EXP_MD = """# 我的经验

## 起手牌
（暂无）

## 翻牌后
（暂无）

## 转牌 / 河牌
（暂无）

## 对手建模
（暂无）

## 注码与位置
（暂无）
"""


EXP_REVISE_INSTRUCTION = """
请基于本手的复盘 + 最近若干手的流水 + 现行经验，决定是否修订经验文档。

修订原则：
1. 经验是可迁移到未来类似情形的策略性判断
2. 优先增量编辑（在原章节下增删条目），不要整段重写
3. 保持五章节结构不变

【Calibration】请显式对比"你 decide 时的 intent"与"实际 outcome"：
- 若 intent 与 outcome 高度一致, 不修订或微调
- 若 intent 与 outcome 出现明显偏差（你以为对手会 fold 但 call 了；你以为是 bluff catch 但被 nut 击败），应在 calibration_note 中说明偏差类型，并据此修订相关章节

【Self-Check】写入新经验前，请检查：你的新经验是否与已知事实冲突（若提供了"过往相关事实"段）？
- 若冲突，是新经验错了？还是该事实是噪声样本？
- 把判断与调和过程写入 self_check 字段
- 若无 fact 段或无冲突，self_check 写 "no conflict found"

输出严格 JSON:
{
  "keep": bool,
  "new_md": "若 keep=false 则填修订后的完整 md，五章节",
  "calibration_note": "intent vs outcome 的偏差说明",
  "self_check": "与已知事实的一致性检查",
  "supporting_fact_ids": ["本次修订所依据的事实 id 列表"]
}
"""


class ExprAgent(BasePlayer):
    def __init__(self, player_id, model_name, starting_stack, output_dir, traj_window=30):
        super().__init__(player_id, model_name, starting_stack, output_dir)
        self.traj_window = traj_window

        self.working_buffer = []
        self.exp_md_path = f"{output_dir}/experience.md"
        self.exp_log_path = f"{output_dir}/experience_log.jsonl"
        self.exp_log = JSONLStore(self.exp_log_path)
        p = Path(self.exp_md_path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(INITIAL_EXP_MD, encoding="utf-8")

        self.action_log = JSONLStore(f"{output_dir}/actions.jsonl")  # for debug
        self.trajectory_log_path = f"{output_dir}/trajectory_log.jsonl"
        self.trajectory_log = JSONLStore(self.trajectory_log_path)  # for debug

    def _build_memory(self):
        """ memory """
        p = Path(self.exp_md_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return INITIAL_EXP_MD

    def _revise_experience(self, hand_index, reflect_prompt, extra_fact_section=None):
        p = Path(self.exp_md_path)
        current_exp_md = p.read_text(encoding="utf-8")

        store = JSONLStore(self.trajectory_log_path)
        rows = store.read_all()
        recent_traj = rows[-self.traj_window:]

        recent_text = "\n\n".join(
            f"--- hand {r['hand_index']} (outcome={r['outcome']}, won={r['won_amount']}) ---\n{r['recap']}"
            for r in recent_traj
        )
        pieces = [
            f"现行经验：\n{current_exp_md}",
            f"最近 {len(recent_traj)} 手的流水（含本手）：\n{recent_text}",
        ]
        if extra_fact_section:
            pieces.append(f"过往相关事实：\n{extra_fact_section}")
        pieces.append(EXP_REVISE_INSTRUCTION)
        prompt = "\n\n".join(pieces)

        response = call_llm(self.model_name, [
            {"role": "system", "content": reflect_prompt},
            {"role": "user", "content": prompt},
        ], json_mode=True)
        revision = safe_json(response) or {}
        if revision.get("keep", True):
            return None
        new_md = (revision.get("new_md") or "").strip()
        if not new_md or new_md == current_exp_md:
            return None
        Path(self.exp_md_path).write_text(new_md, encoding="utf-8")
        record = {
            "rev":                    len(self.exp_log.read_all()) + 1,
            "hand_index":             hand_index,
            "old_md":                 current_exp_md,
            "new_md":                 new_md,
            "calibration_note":       revision.get("calibration_note", ""),
            "self_check":             revision.get("self_check", ""),
            "supporting_fact_ids":    revision.get("supporting_fact_ids", []) or [],
            "contradicting_fact_ids": [],
            "noise_fact_ids":         [],
        }
        self.exp_log.append(record)
        return record

    def decide(self, state):
        # get base prompts
        llm_state, action_prompt, _ = build_base_prompts(state)

        # get memory
        memory_sections = {"过往经验": self._build_memory()}

        # llm
        prompt = build_user_prompt(llm_state, extra_sections=memory_sections)
        response = call_llm(self.model_name, [
            {"role": "system", "content": action_prompt},
            {"role": "user", "content": prompt},
        ], json_mode=True)
        parsed, action = parse_response(response, llm_state["legal_actions"])

        # record
        me_id = llm_state["current_player_id"]
        me = next(p for p in llm_state["players"] if p["id"] == me_id)
        self.working_buffer.append({
            "phase":            llm_state["phase"],
            "board":            list(llm_state["community_cards"]),
            "hole":             list(me["hole_cards"]),
            "pot_before":       llm_state["pot"],
            "to_call":          llm_state["current_bet_this_round"] - me["current_bet"],
            "my_stack_before":  me["stack"],
            "opponents_state":  _alive_opponents_snapshot(llm_state, me["id"]),
            "history_so_far":   list(llm_state["action_history"]),
            "parsed":           parsed,
            "action":           action,
        })
        self.action_log.append({
            "hand_index":      state["hand_index"],
            "phase":           state["phase"],
            "action":          action,
            "intent":          (parsed or {}).get("intent"),
            "memory_sections": list(memory_sections.keys()),
            "raw":             response,
        })
        return action

    def observe(self, final_state):
        if not self.working_buffer:
            return
        if self.frozen:
            self.working_buffer = []
            return

        # get base prompts
        llm_final, _, reflect_prompt = build_base_prompts(final_state)

        # summarize
        recap = summarize_hand(self.working_buffer, llm_final, self.player_id)

        # record
        self.trajectory_log.append({
            "hand_index": final_state["hand_index"],
            "recap":      recap,
            "outcome":    _outcome(final_state, self.player_id),
            "won_amount": _won_amount(final_state, self.player_id),
        })
        try:
            self._revise_experience(final_state["hand_index"], reflect_prompt)
        finally:
            self.working_buffer = []
