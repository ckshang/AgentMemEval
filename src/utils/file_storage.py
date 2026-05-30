import json
import numpy as np

from pathlib import Path


def _alive_opponents_snapshot(state, my_id):
    """ 落进 fact 的对手快照。用相对位置匿名标签 下家1/下家2... 替换真实 id，
    保证未来从 fact 召回的内容里不会出现具体玩家身份。 """
    players = state["players"]
    n = len(players)
    me_idx = next(i for i, p in enumerate(players) if p["id"] == my_id)
    label = {}
    for k in range(1, n):
        label[players[(me_idx + k) % n]["id"]] = f"下家{k}"

    out = []
    for p in players:
        if p["id"] == my_id or p["folded"] or p["busted"]:
            continue
        out.append({
            "label": label.get(p["id"], "?"),
            "stack": p["stack"],
            "current_bet": p["current_bet"],
            "total_committed": p["total_committed"],
        })
    return out


def safe_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


class JSONLStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self):
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class EmbeddingStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def add(self, fact_id, vec):
        self.data[fact_id] = np.asarray(vec, dtype=np.float32)

    def get(self, fact_id):
        return self.data.get(fact_id)

    def save(self):
        # 用 .npy 结尾的临时路径，避免 np.save 自动追加后缀导致路径漂移
        tmp = self.path.with_suffix(self.path.suffix + ".tmp.npy")
        np.save(tmp, self.data, allow_pickle=True)
        tmp.replace(self.path)

    def _load(self):
        if not self.path.exists():
            return {}
        try:
            return np.load(self.path, allow_pickle=True).item()
        except Exception:
            return {}
