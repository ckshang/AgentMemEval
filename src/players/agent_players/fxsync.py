"""
FactExprSyncAgent，事实+经验并行型 memory (事实与经验互不干扰)。
"""
from src.players.agent_players.expr import ExprAgent
from src.players.agent_players.fact import extract_facts_from_buffer
from src.players.prompts import build_base_prompts, build_user_prompt, summarize_hand
from src.players.prompts import _outcome, _won_amount
from src.players.rag import build_retrieval_query, topk_by_similarity, embed
from src.players.llms import call_llm, parse_response
from src.utils.file_storage import JSONLStore, EmbeddingStore, _alive_opponents_snapshot


class FactExprSyncAgent(ExprAgent):
    def __init__(self, player_id, model_name, starting_stack, output_dir, top_k_retrieval=15, traj_window=30):
        super().__init__(player_id, model_name, starting_stack, output_dir, traj_window)
        self.top_k_retrieval = top_k_retrieval

        self.facts_store = JSONLStore(f"{output_dir}/facts.jsonl")
        self._emb_store = EmbeddingStore(f"{output_dir}/fact_embeddings.npy")

    def _build_fact_memory(self, llm_state):
        """ fact memory """
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

    def _build_expr_memory(self):
        """ expr memory """
        return self._build_memory()

    def decide(self, state):
        # get base prompts
        llm_state, action_prompt, _ = build_base_prompts(state)

        # get memory
        memory_sections = {
            "过往相关事实": self._build_fact_memory(llm_state),
            "过往经验": self._build_expr_memory(),
        }

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
            # update fact memory
            new_facts = extract_facts_from_buffer(self.working_buffer, final_state, self.player_id)
            for f in new_facts:
                self.facts_store.append(f)
                self._emb_store.add(f["id"], embed(f["text"])[0])
            self._emb_store.save()

            # update expr memory
            # recent_fact_text = (
            #     "\n".join(f"- {f['text']}" for f in new_facts[-3:]) if new_facts else None
            # )
            # self._revise_experience(recap, final_state["hand_index"], reflect_prompt, extra_fact_section=recent_fact_text)
            self._revise_experience(final_state["hand_index"], reflect_prompt)
        finally:
            self.working_buffer = []