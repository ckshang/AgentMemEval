"""
FactExprAsyncAgent，事实+经验协同型 memory。
架构：事实层 <-> 巩固层 <-> 经验层
"""
import math
import json
import random

from pathlib import Path
from collections import defaultdict

from src.players.agent_players.expr import ExprAgent
from src.players.agent_players.fact import extract_facts_from_buffer
from src.players.prompts import build_base_prompts, build_user_prompt, summarize_hand
from src.players.prompts import _outcome, _won_amount
from src.players.rag import build_retrieval_query, topk_by_similarity, embed
from src.players.llms import call_llm, parse_response
from src.utils.file_storage import JSONLStore, EmbeddingStore, _alive_opponents_snapshot, safe_json


SWEEP_GENERALIZE_INSTRUCTION = """
请基于以下材料修订经验文档：

(1) 最近召回的事实（按召回路径分组，附 fact id）
(2) 自上次 sweep 以来的流水
(3) 现行经验

修订原则：
- 优先增量编辑，保持五章节结构
- 默认对召回事实"不表态"；仅当对某事实有强烈判断时才显式列入对应 ids
- 若发现某条事实与经验冲突 → contradicting
- 若某条事实显然是噪声（一次性偶然结果、对手明显失误等不可复用）→ noise
- 若多条事实指向同一规律，提炼到经验文档对应章节 → supporting

【Calibration】对比预期 vs 实际
【Self-Check】新经验是否与召回事实冲突

输出严格 JSON:
{
  "keep": bool,
  "new_md": "...",
  "calibration_note": "...",
  "self_check": "...",
  "supporting_fact_ids": ["f_xxx", "..."],
  "contradicting_fact_ids": ["f_yyy", "..."],
  "noise_fact_ids": ["f_zzz", "..."]
}
"""


class FactExprAsyncAgent(ExprAgent):
    def __init__(self, player_id, model_name, starting_stack, output_dir, traj_window=30,
                 salience_threshold=0.03, salience_mirror_threshold=0.3, stability_min=0.5, stability_max=50.0, stability_init=10.0,
                 top_k_main=14, top_k_mirror=5, mirror_prob=0.3, sweep_every=5):
        super().__init__(player_id, model_name, starting_stack, output_dir, traj_window)
        self.salience_threshold = salience_threshold
        self.salience_mirror_threshold = salience_mirror_threshold
        self.stability_min = stability_min
        self.stability_max = stability_max
        self.stability_init = stability_init
        self.top_k_main = top_k_main
        self.top_k_mirror = top_k_mirror
        self.mirror_prob = mirror_prob
        self.sweep_every = sweep_every

        # 事实层
        self.facts_store = JSONLStore(f"{output_dir}/facts.jsonl")
        self._emb_store = EmbeddingStore(f"{output_dir}/fact_embeddings.npy")
        self._state_path = Path(f"{output_dir}/facts_state.jsonl")
        self._state_table = self._load_state_table()

        # 巩固层
        self.sweep_log = JSONLStore(f"{output_dir}/sweep_log.jsonl")
        self._sweep_counter = 0

    def _load_state_table(self):
        if not self._state_path.exists():
            return {}
        out = {}
        with open(self._state_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                out[rec["id"]] = {
                    "stability":          float(rec["stability"]),
                    "last_accessed_hand": int(rec["last_accessed_hand"]),
                    "access_count":       int(rec["access_count"]),
                    "linked_exp_revs":    list(rec.get("linked_exp_revs", [])),
                }
        return out

    def _salience(self, fact_id, t):
        """ salience (i.e. 记忆曲线) """
        st = self._state_table.get(fact_id)
        if st is None:
            return 0.0
        gap = max(0, t - st["last_accessed_hand"])
        s = max(self.stability_min, float(st["stability"]))
        return math.exp(-gap / s)

    def _build_fact_memory(self, llm_state):
        """ fact memory (different from factagent) """
        all_facts = self.facts_store.read_all()
        t = llm_state["hand_index"]
        active = [f for f in all_facts if self._salience(f["id"], t) >= self.salience_threshold]
        if not active:
            return "（无）"

        query = build_retrieval_query(llm_state)
        # 主路召回：在 active 集合上做带 salience 权重的相似检索
        top_main, _ = topk_by_similarity(
            query, active, k=self.top_k_main,
            vec_lookup=self._emb_store.data,
            salience_fn=lambda fid: self._salience(fid, t),
        )
        retrieved = list(top_main)
        top_main_ids = {f["id"] for f in top_main}

        # 镜像召回：从"已落到 [salience_threshold, salience_mirror_threshold) 区间"的边缘事实里以 mirror_prob 抽一条，注入偶发跨域记忆
        # 冻结时（评估期）不做随机 mirror，保证泛化桌可复现
        if not self.frozen and random.random() < self.mirror_prob:
            pool = [
                f for f in all_facts
                if f["id"] not in top_main_ids
                   and self.salience_threshold <= self._salience(f["id"], t) < self.salience_mirror_threshold
            ]
            if pool:
                mirror_top, _ = topk_by_similarity(
                    query, pool, k=self.top_k_mirror,
                    vec_lookup=self._emb_store.data, salience_fn=None,
                )
                if mirror_top:
                    retrieved.append(random.choice(mirror_top))

        # 冻结时不更新 access / last_accessed_hand（评估期 agent 不再演化）
        if not self.frozen:
            for f in retrieved:
                st = self._state_table.get(f["id"])
                if st is not None:
                    st["last_accessed_hand"] = t
                    st["access_count"] += 1

        return "\n".join(f"- {f['text']}" for f in retrieved) or "（无）"

    def _build_expr_memory(self):
        """ expr memory """
        return self._build_memory()

    def _sweep(self, final_state, reflect_prompt):
        t = final_state["hand_index"]
        all_facts = self.facts_store.read_all()
        active = [f for f in all_facts if self._salience(f["id"], t) >= self.salience_threshold]

        # 最近 trajectory（作为 prompt 的"近况"）
        rows = self.trajectory_log.read_all()
        recent = rows[-self.sweep_every:]

        # Path-1 的 query：用本手最后一次 decision snapshot 拼，与 decide 路径同口径。
        # final_state 此时已 hand_over、pot 归零，分布和 decide 时差异较大，会让相似召回偏弱。
        if self.working_buffer:
            last = self.working_buffer[-1]
            query_text = (
                f"phase={last['phase']} "
                f"hole={last['hole']} "
                f"board={last['board']} "
                f"pot={last['pot_before']} "
                f"to_call={last['to_call']}"
            )
        else:
            # 兜底：本手没有任何 decision（理论极罕见），退回 final_state
            query_text = build_retrieval_query(final_state, me_id=self.player_id)

        # ========== Step 1: 三路召回 ==========
        # Path 1 — 相似召回（以本手局面为 query，结构化短串）
        # 把和最近相关的一些事实召回，总结经验
        if active:
            p1, _ = topk_by_similarity(query_text, active, k=20, vec_lookup=self._emb_store.data, salience_fn=None)
        else:
            p1 = []
        # Path 2 — 多样性（phase × outcome 分箱）
        # 保持多样性
        buckets = defaultdict(list)
        for f in active:
            buckets[(f.get("final_phase"), f["hand_outcome"])].append(f)
        p2 = []
        for items in buckets.values():
            p2.extend(random.sample(items, k=min(2, len(items))))
        # Path 3 — 重要性
        # 保持显著性
        p3 = sorted(active, key=lambda f: self._salience(f["id"], t), reverse=True)[:10]
        p1, p2, p3 = list(p1), list(p2), list(p3)

        merged = {}
        for f in p1 + p2 + p3:
            merged.setdefault(f["id"], f)
        recalled = random.sample(list(merged.values()), k=min(30, len(merged)))

        # ========== Step 2: Generalize (LLM) ==========
        # 给 LLM 只看 recalled，但按"首次来源 path"归组，保留路径归属信息
        p1_ids = {f["id"] for f in p1}
        p2_ids = {f["id"] for f in p2}
        groups = {"Path-1: 相似召回": [], "Path-2: 多样性召回": [], "Path-3: 重要性召回": []}
        for f in recalled:
            fid = f["id"]
            if fid in p1_ids:
                groups["Path-1: 相似召回"].append(f)
            elif fid in p2_ids:
                groups["Path-2: 多样性召回"].append(f)
            else:
                groups["Path-3: 重要性召回"].append(f)

        def fmt(group, title):
            if not group:
                return f"[{title}]\n（无）"
            lines = [f"[{title}]"]
            for f in group:
                lines.append(f"- ({f['id']}, h{f['hand_index']}, {f.get('final_phase')}, {f['hand_outcome']}) {f['text']}")
            return "\n".join(lines)

        recalled_text = "\n\n".join(fmt(g, title) for title, g in groups.items())
        recent_traj_text = "\n\n".join(
            f"--- hand {r['hand_index']} (outcome={r['outcome']}, won={r['won_amount']}) ---\n{r['recap']}"
            for r in recent
        ) or "（无）"
        current_exp_md = self._build_expr_memory()

        # prompt 顺序：先看到最近发生了什么，再回忆相关事实，最后看现行经验
        prompt = (
            f"(1) 自上次 sweep 的近况（最近 {len(recent)} 手流水）:\n{recent_traj_text}\n\n"
            f"(2) 由上述近况触发回忆起的事实:\n{recalled_text}\n\n"
            f"(3) 现行经验:\n{current_exp_md}\n\n"
            f"{SWEEP_GENERALIZE_INSTRUCTION}"
        )
        response = call_llm(self.model_name, [
            {"role": "system", "content": reflect_prompt},
            {"role": "user",   "content": prompt},
        ], json_mode=True)
        revision = safe_json(response)

        rev_num = None
        sup_ids, con_ids, noi_ids = [], [], []
        if revision:
            sup_ids = list(revision.get("supporting_fact_ids") or [])
            con_ids = list(revision.get("contradicting_fact_ids") or [])
            noi_ids = list(revision.get("noise_fact_ids") or [])
            # 即使 keep=False 但 new_md 为空/未变，也只跳过写 experience.md，
            # 不阻断 Step 3 的 stability reweight 和 sweep_log（fact ids 仍有价值）。
            if not revision.get("keep", True):
                new_md = (revision.get("new_md") or "").strip()
                if new_md and new_md != current_exp_md:
                    Path(self.exp_md_path).write_text(new_md, encoding="utf-8")
                    rev_num = len(self.exp_log.read_all()) + 1
                    self.exp_log.append({
                        "rev": rev_num,
                        "hand_index": final_state["hand_index"],
                        "old_md": current_exp_md,
                        "new_md": new_md,
                        "calibration_note": revision.get("calibration_note", ""),
                        "self_check": revision.get("self_check", ""),
                        "supporting_fact_ids": list(revision.get("supporting_fact_ids") or []),
                        "contradicting_fact_ids": list(revision.get("contradicting_fact_ids") or []),
                        "noise_fact_ids": list(revision.get("noise_fact_ids") or []),
                    })

        # ========== Step 3: Reweight ==========
        sup_set, con_set, noi_set = set(sup_ids), set(con_ids), set(noi_ids)
        for fid, st in self._state_table.items():
            s = float(st["stability"])
            if fid in noi_set:
                st["stability"] = max(s * 0.3, self.stability_min)
            elif fid in con_set:
                st["stability"] = min(s * 2.0, self.stability_max)
            elif fid in sup_set:
                st["stability"] = min(s * 1.5, self.stability_max)

        if rev_num is not None:
            for fid in (sup_ids + con_ids + noi_ids):
                if fid in self._state_table:
                    self._state_table[fid]["linked_exp_revs"].append(rev_num)

        self.sweep_log.append({
            "sweep_idx":      len(self.sweep_log.read_all()) + 1,
            "hand_index":     t,
            "n_recalled":     len(recalled),
            "n_supporting":   len(sup_ids),
            "n_contradicting": len(con_ids),
            "n_noise":        len(noi_ids),
            "rev_created":    rev_num,
        })

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
            def _save_state_table():
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    for fid, st in self._state_table.items():
                        rec = {"id": fid, **st}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tmp.replace(self._state_path)

            # update fact memory
            new_facts = extract_facts_from_buffer(self.working_buffer, final_state, self.player_id)
            for f in new_facts:
                self.facts_store.append(f)
                self._emb_store.add(f["id"], embed(f["text"])[0])
                self._state_table[f["id"]] = {
                    "stability": self.stability_init,
                    "last_accessed_hand": final_state["hand_index"],
                    "access_count": 0,
                    "linked_exp_revs": [],
                }
            self._emb_store.save()
            _save_state_table()

            # update expr memory
            self._sweep_counter += 1
            if self._sweep_counter >= self.sweep_every:
                try:
                    self._sweep(final_state, reflect_prompt)
                finally:
                    self._sweep_counter = 0
                    _save_state_table()
        finally:
            self.working_buffer = []