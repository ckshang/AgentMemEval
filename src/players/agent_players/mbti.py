"""
MBTIAgent，在 fxsync（fact RAG + 每手 revise experience）底座上，
把读侧/写侧两个"选择器"换成带 MBTI 人格的独立 LLM 裁决层。

- 读侧（decide 前）：_persona_curate —— 用人格 LLM 把召回的 fact + experience
  筛选/重排成精简记忆视图，再喂给中性的决策 LLM。
- 写侧（observe 后）：复用父类 _revise_experience，但传入人格化 reflect prompt。
  fact 抽取保持中性、机械。
"""
from src.players.agent_players.fxsync import FactExprSyncAgent
from src.players.agent_players.fact import extract_facts_from_buffer
from src.players.personas import PERSONAS, build_action_prompt, build_curate_prompt, build_reflect_prompt
from src.players.prompts import build_base_prompts, build_user_prompt, summarize_hand
from src.players.prompts import _outcome, _won_amount
from src.players.rag import embed
from src.players.llms import call_llm, parse_response
from src.utils.file_storage import _alive_opponents_snapshot, safe_json


class MBTIAgent(FactExprSyncAgent):
    def __init__(self, player_id, model_name, starting_stack, output_dir,
                 persona, top_k_retrieval=15, traj_window=30):
        super().__init__(player_id, model_name, starting_stack, output_dir,
                         top_k_retrieval=top_k_retrieval, traj_window=traj_window)
        if persona not in PERSONAS:
            raise ValueError(f"unknown MBTI persona: {persona}")
        self.persona = persona
        self._persona_text = PERSONAS[persona]
        self.action_prompt = build_action_prompt(self._persona_text)
        self.curate_prompt = build_curate_prompt(self._persona_text)
        self.reflect_prompt = build_reflect_prompt(self._persona_text)

    def _persona_curate(self, raw_fact, raw_expr):
        """读侧裁决层：人格 LLM 筛选/重组记忆。失败兜底为原始拼接（退化中性）。"""
        fallback = f"【过往相关事实】\n{raw_fact}\n\n【现行经验】\n{raw_expr}"
        user_msg = (
            "请筛选以下记忆：\n\n"
            f"=== 过往相关事实 ===\n{raw_fact}\n\n"
            f"=== 现行经验 ===\n{raw_expr}"
        )
        try:
            response = call_llm(self.model_name, [
                {"role": "system", "content": self.curate_prompt},
                {"role": "user", "content": user_msg},
            ], json_mode=True)
            curated = (safe_json(response) or {}).get("curated")
            if curated and isinstance(curated, str) and curated.strip():
                return curated.strip()
        except Exception:
            pass
        return fallback

    def decide(self, state):
        # base prompts（决策侧 system prompt 用人格版，不用中性返回值）
        llm_state, _, _ = build_base_prompts(state)

        # raw memory（复用父类两条召回路径）
        raw_fact = self._build_fact_memory(llm_state)
        raw_expr = self._build_expr_memory()

        # 读侧裁决层（人格 LLM）
        filtered = self._persona_curate(raw_fact, raw_expr)
        memory_sections = {"经人格筛选的记忆": filtered}

        # 决策 LLM（人格化）
        prompt = build_user_prompt(llm_state, extra_sections=memory_sections)
        response = call_llm(self.model_name, [
            {"role": "system", "content": self.action_prompt},
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
            "persona":         self.persona,
            "raw_fact_len":    len(raw_fact),
            "raw_expr_len":    len(raw_expr),
            "filtered_len":    len(filtered),
            "raw":             response,
        })
        return action

    def observe(self, final_state):
        if not self.working_buffer:
            return
        if self.frozen:
            self.working_buffer = []
            return

        # base prompts（reflect_prompt 用人格版，不用中性返回值）
        llm_final, _, _ = build_base_prompts(final_state)

        # summarize
        recap = summarize_hand(self.working_buffer, llm_final, self.player_id)
        self.trajectory_log.append({
            "hand_index": final_state["hand_index"],
            "recap":      recap,
            "outcome":    _outcome(final_state, self.player_id),
            "won_amount": _won_amount(final_state, self.player_id),
        })
        try:
            # fact 抽取保持中性
            new_facts = extract_facts_from_buffer(self.working_buffer, final_state, self.player_id)
            for f in new_facts:
                self.facts_store.append(f)
                self._emb_store.add(f["id"], embed(f["text"])[0])
            self._emb_store.save()

            # 写侧裁决层：人格化 reflect prompt
            self._revise_experience(final_state["hand_index"], self.reflect_prompt)
        finally:
            self.working_buffer = []
