"""
FactAgent，事实型 memory。
"""
import uuid

from src.players.base_player import BasePlayer
from src.players.prompts import build_base_prompts, build_user_prompt, summarize_hand
from src.players.prompts import _outcome, _won_amount, _fmt_action
from src.players.rag import build_retrieval_query, topk_by_similarity, embed
from src.players.llms import call_llm, parse_response
from src.utils.file_storage import JSONLStore, EmbeddingStore, _alive_opponents_snapshot


def render_fact_text(fact):
    """ 把一条 fact dict 渲染为 embedding/LLM 可读的文本视图。 """
    board = " ".join(fact["board"]) if fact["board"] else "(空)"
    hole = " ".join(fact["hole"]) if fact["hole"] else "(空)"
    hist = " | ".join(
        f"{h['phase']} {h['player_id']} {_fmt_action(h['action'])}"
        for h in fact.get("history_so_far", [])
    ) or "(无)"
    intent = fact.get("my_intent") or "(无)"
    action_str = _fmt_action(fact.get("my_action") or {})
    outcome = fact.get("hand_outcome")
    won = fact.get("hand_won_amount", 0)
    return (
        f"[hand {fact['hand_index']}, decision #{fact['decision_idx']}, {fact['phase']}]\n"
        f"board={board}, hole={hole}, pot={fact['pot_before']}, to_call={fact['to_call']}\n"
        f"history: {hist}\n"
        f"my_intent: {intent}\n"
        f"my_action: {action_str}\n"
        f"hand_outcome: {outcome} {won}"
    )


def extract_facts_from_buffer(working_buffer, final_state, my_id):
    """ 从 working_buffer 确定性抽取 fact 列表。 """
    hand_idx = final_state["hand_index"]
    outcome = _outcome(final_state, my_id)
    won = _won_amount(final_state, my_id)
    facts = []
    for i, turn in enumerate(working_buffer):
        parsed = turn.get("parsed") or {}
        fact = {
            "id":               "f_" + uuid.uuid4().hex[:8],
            "hand_index":       hand_idx,
            "decision_idx":     i,
            "phase":            turn["phase"],
            "board":            turn["board"],
            "hole":             turn["hole"],
            "pot_before":       turn["pot_before"],
            "to_call":          turn["to_call"],
            "my_stack_before":  turn["my_stack_before"],
            "opponents_state":  turn["opponents_state"],
            "history_so_far":   turn["history_so_far"],
            "my_intent":        parsed.get("intent"),
            "my_action":        turn["action"],
            "hand_outcome":     outcome,
            "hand_won_amount":  won,
        }
        fact["text"] = render_fact_text(fact)
        facts.append(fact)
    return facts


class FactAgent(BasePlayer):
    def __init__(self, player_id, model_name, starting_stack, output_dir, top_k_retrieval=30):
        super().__init__(player_id, model_name, starting_stack, output_dir)
        self.top_k_retrieval = top_k_retrieval

        self.working_buffer = []
        self.facts_store = JSONLStore(f"{output_dir}/facts.jsonl")
        self._emb_store = EmbeddingStore(f"{output_dir}/fact_embeddings.npy")

        self.action_log = JSONLStore(f"{output_dir}/actions.jsonl")  # for debug
        self.trajectory_log = JSONLStore(f"{output_dir}/trajectory_log.jsonl")  # for debug

    def _build_memory(self, llm_state):
        """ memory """
        all_facts = self.facts_store.read_all()
        if not all_facts:
            return "（无）"
        else:
            query = build_retrieval_query(llm_state)
            top_facts, _ = topk_by_similarity(
                query, all_facts, k=self.top_k_retrieval,
                vec_lookup=self._emb_store.data,
                salience_fn=None,
            )
            return "\n".join(f"- {f['text']}" for f in top_facts) or "（无）"

    def decide(self, state):
        # get base prompts
        llm_state, action_prompt, _ = build_base_prompts(state)

        # get memory
        memory_sections = {"过往相关事实": self._build_memory(llm_state)}

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
        llm_final, _, _ = build_base_prompts(final_state)

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
            new_facts = extract_facts_from_buffer(self.working_buffer, final_state, self.player_id)
            for f in new_facts:
                self.facts_store.append(f)
                self._emb_store.add(f["id"], embed(f["text"])[0])
            self._emb_store.save()
        finally:
            self.working_buffer = []

    def extract_methodology(self):
        facts = self.facts_store.read_all()
        if not facts:
            return "## 暂无事实"
        else:
            parts = [f"## 共 {len(facts)} 条事实（流水型, 按时间）\n"]
            for f in facts[-30:]:
                parts.append(f"- (h{f['hand_index']}#{f['decision_idx']}, {f['phase']}, "
                             f"intent={f.get('my_intent')!r}, outcome={f['hand_outcome']})")
            return "\n".join(parts)