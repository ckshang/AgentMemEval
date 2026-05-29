"""
exp1_memory 专用评估画图。
- 主桌 avg stack by algo
- 泛化 avg stack by algo
- 诈唬率 by algo（两种定义：proxy 基于结果，intent 基于 LLM 内心活动）
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_BLUFF_KEYWORDS = ["bluff", "诈唬", "虚张", "唬人", "诈"]


def _algo_of_pid(pid):
    """player_00_fact-deepseekv4f -> fact ；player_00_random -> random"""
    parts = pid.split("_", 2)
    if len(parts) < 3:
        return pid
    tail = parts[2]
    if tail in ("random", "naive"):
        return tail
    return tail.split("-")[0]


# ============================================================
# 1. 主桌：每种算法的平均 stack 曲线
# ============================================================

def plot_avg_stack_by_algo(output_dir, save_path=None, ax=None):
    stacks_path = Path(output_dir) / "stacks.csv"
    if not stacks_path.exists():
        raise FileNotFoundError(f"missing {stacks_path}")
    df = pd.read_csv(stacks_path)
    df["algo"] = df["player_id"].map(_algo_of_pid)
    avg = df.groupby(["algo", "hand_index"])["stack"].mean().reset_index()

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for algo, sub in avg.groupby("algo"):
        sub = sub.sort_values("hand_index")
        ax.plot(sub["hand_index"], sub["stack"], marker="o", markersize=3, label=algo)
    ax.set_xlabel("Hand Index")
    ax.set_ylabel("Avg Stack per Algo")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.tight_layout(); plt.savefig(save_path, dpi=120)
    return ax


# ============================================================
# 2. 泛化：每个 snapshot → 5 个 NaiveLLM 桌 10 手；每个 hand_index 算法平均 stack
# ============================================================

def plot_generalization_by_algo(generalization_results, save_path=None, ax=None):
    """generalization_results: list[dict] of
        {"checkpoint_hand": int, "algo": "fact", "src_pid": "player_00_fact-...",
         "rows": [{"hand_index": int, "stacks": {pid -> stack}, "main_pid": ...}, ...]}

    画图：x = "main_hand_index" = checkpoint_hand + sub_hand 偏移；y = main agent stack；
    同一 algo 多个起点 + 多个 hand 求 mean。
    """
    flat = []
    for entry in generalization_results:
        algo = entry["algo"]
        cp = entry["checkpoint_hand"]
        for r in entry["rows"]:
            stack = r["stacks"].get(r["main_pid"])
            if stack is None:
                continue
            flat.append({
                "algo": algo,
                "checkpoint": cp,
                "sub_hand": r["hand_index"],
                "stack": stack,
            })
    if not flat:
        print("[plot_generalization_by_algo] 无数据")
        return None
    df = pd.DataFrame(flat)

    # 每个 (algo, sub_hand) 在多 checkpoint 上求均值
    agg = df.groupby(["algo", "sub_hand"])["stack"].mean().reset_index()

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for algo, sub in agg.groupby("algo"):
        sub = sub.sort_values("sub_hand")
        ax.plot(sub["sub_hand"], sub["stack"], marker="o", markersize=4, label=algo)
    ax.set_xlabel("Hand Index")
    ax.set_ylabel("Avg Stack per Algo (across snapshots)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.tight_layout(); plt.savefig(save_path, dpi=120)
    return ax


# ============================================================
# 3. 诈唬率
# ============================================================
# 诈唬定义（精确版，需 hands.csv 含 showdown_ranks 列）：
#   - 该手走到摊牌（went_to_showdown=True），且
#   - 该 pid 在 postflop（flop/turn/river）至少主动 raise 过一次，且
#   - 摊牌时该 pid 的 hand_rank 为 "High Card"
# 分母：满足前两条的手数 (postflop 主动 raise + 摊牌)。
# 分子：上述并且 hand_rank 为 High Card 的手数。
# 排除项：preflop-only raise、未摊牌就被对手 fold 收掉的手。

_HIGH_CARD_RE = re.compile(r"high card", re.I)
_POSTFLOP_PHASES = {"flop", "turn", "river"}


def _parse_showdown_ranks(s):
    """ "pid1=High Card;pid2=Pair" -> {"pid1": "High Card", "pid2": "Pair"} """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return {}
    out = {}
    for kv in str(s).split(";"):
        if "=" in kv:
            pid, rk = kv.split("=", 1)
            out[pid.strip()] = rk.strip()
    return out


def _rolling_rate_by_algo(df, window):
    """ df 列: hand_index, pid, qualifies, is_bluff, algo。
        返回 dict[algo] -> (hand_indices, rolling_rate)。 """
    series = {}
    for algo, sub in df.groupby("algo"):
        per_hand = sub.groupby("hand_index").agg(
            qualifies=("qualifies", "sum"),
            bluffs=("is_bluff", "sum"),
        ).sort_index()
        roll_q = per_hand["qualifies"].rolling(window=window, min_periods=1).sum()
        roll_b = per_hand["bluffs"].rolling(window=window, min_periods=1).sum()
        rate = (roll_b / roll_q.replace(0, np.nan)).fillna(0.0)
        series[algo] = (per_hand.index.values, rate.values)
    return series


def compute_bluff_rate_series_proxy(output_dir, window=10):
    """ Proxy 诈唬率（基于结果）：
          分母 = 走到摊牌 + 翻后主动 raise 过的"手 × pid"
          分子 = 上述并且摊牌牌型为 High Card
        返回 dict[algo] -> (hand_indices, rolling_bluff_rate)。 """
    out = Path(output_dir)
    actions = pd.read_csv(out / "actions.csv")
    hands = pd.read_csv(out / "hands.csv").sort_values("hand_index").reset_index(drop=True)
    if "showdown_ranks" not in hands.columns:
        raise ValueError(
            "hands.csv 缺少 showdown_ranks 列；该列由 game_logger 写入，"
            "需要用新版 controller/game_logger 重新跑一次主桌。"
        )

    pids = sorted(actions["player_id"].unique())

    # 短码 all-in 不构成 effective_raise，不算 postflop 主动 raise。
    # 老数据没有该列时退化为"任何 raise 都算"。
    if "effective_raise" in actions.columns:
        eff_mask = pd.to_numeric(actions["effective_raise"], errors="coerce").fillna(0) > 0
    else:
        eff_mask = actions["action_type"] == "raise"

    per = []  # (hand_index, pid, qualifies, is_bluff)
    for _, h in hands.iterrows():
        h_idx = int(h["hand_index"])
        went_sd = bool(h["went_to_showdown"])
        ranks = _parse_showdown_ranks(h["showdown_ranks"])
        for pid in pids:
            row_mask = (actions["hand_index"] == h_idx) & (actions["player_id"] == pid)
            sub = actions[row_mask]
            postflop_raise = bool(
                ((sub["action_type"] == "raise")
                 & (sub["phase"].isin(_POSTFLOP_PHASES))
                 & eff_mask[row_mask]).any()
            )
            qualifies = went_sd and postflop_raise and (pid in ranks)
            if not qualifies:
                per.append((h_idx, pid, False, False))
                continue
            is_bluff = bool(_HIGH_CARD_RE.search(ranks[pid] or ""))
            per.append((h_idx, pid, True, is_bluff))

    df = pd.DataFrame(per, columns=["hand_index", "pid", "qualifies", "is_bluff"])
    df["algo"] = df["pid"].map(_algo_of_pid)
    return _rolling_rate_by_algo(df, window=window)


def _intent_has_bluff(intent, keywords):
    if not intent:
        return False
    s = str(intent).lower()
    return any(kw.lower() in s for kw in keywords)


def compute_bluff_rate_series_intent(output_dir, window=10, bluff_keywords=None):
    """ Intent 诈唬率（基于 LLM 内心活动）：
          分母 = 该手 pid 至少做过一次"有效"翻后 raise（短码 all-in 不算）
          分子 = 上述并且这些翻后 raise 的 intent 里出现诈唬关键词
        每个 agent 的 actions.jsonl 含 intent 字段；naive 没有 memory 但也有 intent。
        通过 controller 写的 actions.csv 的 effective_raise 列过滤短码 all-in。 """
    if bluff_keywords is None:
        bluff_keywords = _BLUFF_KEYWORDS

    out = Path(output_dir)
    per = []  # (hand_index, pid, qualifies, is_bluff)

    # 从 actions.csv 取每 (hand_index, pid) 是否有"有效"翻后 raise。
    # 老数据缺列时退化为"任何 raise 都算"。
    eff_lookup = set()
    actions_csv = out / "actions.csv"
    if actions_csv.exists():
        ctrl_actions = pd.read_csv(actions_csv)
        if "effective_raise" in ctrl_actions.columns:
            eff_mask = pd.to_numeric(ctrl_actions["effective_raise"], errors="coerce").fillna(0) > 0
        else:
            eff_mask = ctrl_actions["action_type"] == "raise"
        postflop = (ctrl_actions["action_type"] == "raise") & ctrl_actions["phase"].isin(_POSTFLOP_PHASES) & eff_mask
        for _, row in ctrl_actions[postflop].iterrows():
            eff_lookup.add((int(row["hand_index"]), row["player_id"]))

    for player_dir in out.iterdir():
        if not player_dir.is_dir() or not player_dir.name.startswith("player_"):
            continue
        actions_path = player_dir / "actions.jsonl"
        if not actions_path.exists():
            continue
        pid = player_dir.name

        per_hand = defaultdict(list)
        with open(actions_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                per_hand[int(rec["hand_index"])].append(rec)

        for h_idx, recs in per_hand.items():
            postflop_raises = [
                r for r in recs
                if (r.get("action") or {}).get("type") == "raise"
                   and r.get("phase") in _POSTFLOP_PHASES
            ]
            # 必须该手该 pid 至少有一次"有效" postflop raise 才计入
            qualifies = len(postflop_raises) > 0 and (h_idx, pid) in eff_lookup
            if not qualifies:
                per.append((h_idx, pid, False, False))
                continue
            is_bluff = any(_intent_has_bluff(r.get("intent"), bluff_keywords) for r in postflop_raises)
            per.append((h_idx, pid, True, is_bluff))

    df = pd.DataFrame(per, columns=["hand_index", "pid", "qualifies", "is_bluff"])
    if df.empty:
        return {}
    df["algo"] = df["pid"].map(_algo_of_pid)
    return _rolling_rate_by_algo(df, window=window)


# 保留旧名以兼容外部调用
def compute_bluff_rate_series(output_dir, window=10):
    return compute_bluff_rate_series_proxy(output_dir, window=window)


def plot_bluff_rate_by_algo(output_dir, window=10, save_path=None, axes=None):
    """ 画两张子图：左 = proxy（基于结果），右 = intent（基于 LLM 自述）。 """
    proxy_series = compute_bluff_rate_series_proxy(output_dir, window=window)
    intent_series = compute_bluff_rate_series_intent(output_dir, window=window)

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    ax_p, ax_i = axes

    for algo, (x, y) in proxy_series.items():
        ax_p.plot(x, y, marker=".", markersize=4, label=algo)
    ax_p.set_title(f"Proxy (showdown=High Card | postflop raise + showdown)")
    ax_p.set_xlabel("Hand Index")
    ax_p.set_ylabel(f"Bluff Rate (window={window})")
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.legend(loc="best", fontsize=9)
    ax_p.grid(True, alpha=0.3)

    for algo, (x, y) in intent_series.items():
        ax_i.plot(x, y, marker=".", markersize=4, label=algo)
    ax_i.set_title(f"Intent (intent has bluff keyword | postflop raise)")
    ax_i.set_xlabel("Hand Index")
    ax_i.set_ylim(-0.02, 1.02)
    ax_i.legend(loc="best", fontsize=9)
    ax_i.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout(); plt.savefig(save_path, dpi=120)
    return axes


# ============================================================
# 一键报告
# ============================================================

def report(output_dir, generalization_results=None, save_dir=None, window=10):
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print("=== Plot 1: 主桌 avg stack by algo ===")
    plot_avg_stack_by_algo(
        output_dir,
        save_path=(save_dir / "avg_stack_by_algo.png") if save_dir else None,
    )
    plt.show()

    if generalization_results:
        print("=== Plot 2: 泛化桌 avg stack by algo ===")
        plot_generalization_by_algo(
            generalization_results,
            save_path=(save_dir / "generalization_by_algo.png") if save_dir else None,
        )
        plt.show()
    else:
        print("(跳过泛化图：未提供 generalization_results)")

    print("=== Plot 3: 诈唬率 by algo ===")
    plot_bluff_rate_by_algo(
        output_dir, window=window,
        save_path=(save_dir / "bluff_rate_by_algo.png") if save_dir else None,
    )
    plt.show()
