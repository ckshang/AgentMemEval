"""
exp2_mbti 专用评估：by-MBTI 的 action behavior 与 memory behavior。

复用 memory_eval._algo_of_pid 做分组（MBTI 类型即 algo 标签）。
所有指标都从已落盘文件算，不需重跑：
  - action：output_dir/actions.csv（controller 视角的动作流）
  - memory：各 player_xx/actions.jsonl（含 raw_fact_len/filtered_len）
            各 player_xx/experience_log.jsonl（经验修订记录）
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.eval.memory_eval import _algo_of_pid


_POSTFLOP_PHASES = {"flop", "turn", "river"}


# ============================================================
# A. Action behavior（人格 → 怎么打）
# ============================================================

def compute_action_stats(output_dir):
    """
    从 actions.csv 算每个 pid 的行为画像，再按 MBTI 聚合。
    返回 DataFrame（index=algo），列：
      vpip            主动入池率 = 有 call/raise 的手数 / 总参与手数
      raise_rate      raise 动作 / 总动作
      call_rate       call 动作 / 总动作
      check_rate      check 动作 / 总动作
      fold_rate       fold 动作 / 总动作
      fold_to_raise   面对加注后弃牌的次数 / 面对加注的次数（同手内有人 raise 在前）
      avg_raise_amt   平均 raise 到的总额
    """
    out = Path(output_dir)
    actions = pd.read_csv(out / "actions.csv")
    actions = actions.sort_values(["hand_index", "player_id"]).reset_index(drop=True)

    rows = []
    for pid, sub in actions.groupby("player_id"):
        n_actions = len(sub)
        if n_actions == 0:
            continue
        counts = sub["action_type"].value_counts()
        n_raise = int(counts.get("raise", 0))
        n_call = int(counts.get("call", 0))
        n_check = int(counts.get("check", 0))
        n_fold = int(counts.get("fold", 0))

        # VPIP：本手内是否 voluntarily call/raise（check/fold 不算入池）
        hands_played = sub["hand_index"].nunique()
        hands_vpip = sub[sub["action_type"].isin(["call", "raise"])]["hand_index"].nunique()
        vpip = hands_vpip / hands_played if hands_played else 0.0

        amt = pd.to_numeric(sub.loc[sub["action_type"] == "raise", "action_amount"], errors="coerce")
        avg_raise_amt = float(amt.mean()) if len(amt) else 0.0

        rows.append({
            "player_id": pid,
            "vpip": vpip,
            "raise_rate": n_raise / n_actions,
            "call_rate": n_call / n_actions,
            "check_rate": n_check / n_actions,
            "fold_rate": n_fold / n_actions,
            "avg_raise_amt": avg_raise_amt,
        })

    # fold-to-raise：需要看每手每街的动作顺序——某 pid 行动时，本街此前是否已有人 raise，
    # 若有且该 pid 选择 fold，则计入分子；分母是"该 pid 面对前置 raise 的次数"。
    ftr_faced = {}
    ftr_folded = {}
    for (h, ph), grp in actions.groupby(["hand_index", "phase"]):
        raised = False
        for _, ev in grp.iterrows():
            pid = ev["player_id"]
            if raised:
                ftr_faced[pid] = ftr_faced.get(pid, 0) + 1
                if ev["action_type"] == "fold":
                    ftr_folded[pid] = ftr_folded.get(pid, 0) + 1
            if ev["action_type"] == "raise":
                raised = True

    for r in rows:
        pid = r["player_id"]
        faced = ftr_faced.get(pid, 0)
        r["fold_to_raise"] = (ftr_folded.get(pid, 0) / faced) if faced else 0.0

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["algo"] = df["player_id"].map(_algo_of_pid)
    # 同 MBTI 多个玩家取均值
    return df.groupby("algo").mean(numeric_only=True)


_ACTION_METRIC_ORDER = [
    "vpip", "raise_rate", "call_rate", "check_rate", "fold_rate", "fold_to_raise",
]


def plot_action_stats(output_dir, save_path=None):
    """by-MBTI 行为画像：左 = 比率类指标分组柱状图，右 = 平均 raise 尺度。"""
    stats = compute_action_stats(output_dir)
    if stats.empty:
        print("(无 action 数据)")
        return None

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    metrics = [m for m in _ACTION_METRIC_ORDER if m in stats.columns]
    algos = list(stats.index)
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(algos))
    for i, algo in enumerate(algos):
        vals = [stats.loc[algo, m] for m in metrics]
        ax_l.bar(x + i * width, vals, width, label=algo)
    ax_l.set_xticks(x + width * (len(algos) - 1) / 2)
    ax_l.set_xticklabels(metrics, rotation=20)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Action behavior by MBTI")
    ax_l.legend(fontsize=9)
    ax_l.grid(True, axis="y", alpha=0.3)

    if "avg_raise_amt" in stats.columns:
        ax_r.bar(algos, stats["avg_raise_amt"].values)
        ax_r.set_title("Avg raise-to amount")
        ax_r.set_ylabel("chips")
        ax_r.tick_params(axis="x", rotation=20)
        ax_r.grid(True, axis="y", alpha=0.3)

    if save_path:
        plt.tight_layout(); plt.savefig(save_path, dpi=120)
    return stats


# ============================================================
# B. Memory behavior（人格 → 怎么记 / 怎么用记忆）
# ============================================================

def _iter_player_dirs(output_dir):
    out = Path(output_dir)
    for d in sorted(out.iterdir()):
        if d.is_dir() and d.name.startswith("player_"):
            yield d


def compute_curation_stats(output_dir):
    """
    读侧裁决层行为：从各 player 的 actions.jsonl 读 raw_fact_len/raw_expr_len/filtered_len。
    返回 DataFrame（index=algo），列：
      n_decisions     有裁决记录的决策点数
      avg_raw_len     平均原始记忆长度（fact+expr）
      avg_filtered    平均筛后长度
      compression     filtered / raw（越小=筛得越狠）
    只统计 raw>0 的决策点（早期记忆全空时裁决无意义，排除以免稀释）。
    """
    rows = []
    for pdir in _iter_player_dirs(output_dir):
        ap = pdir / "actions.jsonl"
        if not ap.exists():
            continue
        raws, filts = [], []
        with open(ap, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                raw = (rec.get("raw_fact_len") or 0) + (rec.get("raw_expr_len") or 0)
                filt = rec.get("filtered_len")
                if raw > 0 and filt is not None:
                    raws.append(raw)
                    filts.append(filt)
        if not raws:
            continue
        raws, filts = np.array(raws, float), np.array(filts, float)
        rows.append({
            "player_id": pdir.name,
            "n_decisions": len(raws),
            "avg_raw_len": raws.mean(),
            "avg_filtered": filts.mean(),
            "compression": (filts / np.where(raws == 0, np.nan, raws)).mean(),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["algo"] = df["player_id"].map(_algo_of_pid)
    return df.groupby("algo").mean(numeric_only=True)


def compute_revision_stats(output_dir):
    """
    写侧裁决层行为：从各 player 的 experience_log.jsonl 算经验修订强度。
    返回 DataFrame（index=algo），列：
      n_revisions     修订次数
      avg_delta_chars 平均每次修订的 |len(new_md)-len(old_md)|（净改动幅度的代理）
    """
    rows = []
    for pdir in _iter_player_dirs(output_dir):
        ep = pdir / "experience_log.jsonl"
        if not ep.exists():
            continue
        deltas = []
        with open(ep, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                old = rec.get("old_md") or ""
                new = rec.get("new_md") or ""
                deltas.append(abs(len(new) - len(old)))
        rows.append({
            "player_id": pdir.name,
            "n_revisions": len(deltas),
            "avg_delta_chars": float(np.mean(deltas)) if deltas else 0.0,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["algo"] = df["player_id"].map(_algo_of_pid)
    return df.groupby("algo").mean(numeric_only=True)


def plot_memory_behavior(output_dir, save_path=None):
    """by-MBTI memory behavior：读侧压缩比 + 经验修订频率/幅度。"""
    cur = compute_curation_stats(output_dir)
    rev = compute_revision_stats(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax0, ax1, ax2 = axes

    if not cur.empty:
        ax0.bar(list(cur.index), cur["compression"].values)
        ax0.set_title("Read-side compression (filtered/raw)")
        ax0.set_ylabel("ratio (lower = filters harder)")
        ax0.tick_params(axis="x", rotation=20)
        ax0.grid(True, axis="y", alpha=0.3)
    else:
        ax0.set_title("(no curation data)")

    if not rev.empty:
        ax1.bar(list(rev.index), rev["n_revisions"].values)
        ax1.set_title("Experience revisions (count)")
        ax1.set_ylabel("revisions")
        ax1.tick_params(axis="x", rotation=20)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2.bar(list(rev.index), rev["avg_delta_chars"].values)
        ax2.set_title("Avg revision magnitude")
        ax2.set_ylabel("|Δ chars|")
        ax2.tick_params(axis="x", rotation=20)
        ax2.grid(True, axis="y", alpha=0.3)
    else:
        ax1.set_title("(no revision data)")
        ax2.set_title("(no revision data)")

    if save_path:
        plt.tight_layout(); plt.savefig(save_path, dpi=120)
    return cur, rev


# ============================================================
# 一键报告（exp2 专用）
# ============================================================

def report(output_dir, save_dir=None):
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print("=== Action behavior by MBTI ===")
    stats = plot_action_stats(
        output_dir,
        save_path=(save_dir / "action_behavior_by_mbti.png") if save_dir else None,
    )
    plt.show()
    if stats is not None and not stats.empty:
        print(stats.round(3).to_string())
        if save_dir:
            stats.round(4).to_csv(save_dir / "action_behavior_by_mbti.csv")

    print("\n=== Memory behavior by MBTI ===")
    cur, rev = plot_memory_behavior(
        output_dir,
        save_path=(save_dir / "memory_behavior_by_mbti.png") if save_dir else None,
    )
    plt.show()
    if not cur.empty:
        print("[read-side curation]")
        print(cur.round(3).to_string())
        if save_dir:
            cur.round(4).to_csv(save_dir / "curation_by_mbti.csv")
    if not rev.empty:
        print("[experience revision]")
        print(rev.round(3).to_string())
        if save_dir:
            rev.round(4).to_csv(save_dir / "revision_by_mbti.csv")
