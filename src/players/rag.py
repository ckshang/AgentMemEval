import numpy as np

from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer


_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        model_dir = snapshot_download("BAAI/bge-small-zh-v1.5")
        _embedder = SentenceTransformer(model_dir)
    return _embedder


def embed(texts):
  if isinstance(texts, str):
      texts = [texts]
  return np.asarray(_get_embedder().encode(texts, normalize_embeddings=True), dtype=np.float32)


def build_retrieval_query(llm_state, me_id=None):
    """ 生成 retrieval query。仅状态特征，不含策略词。
    me_id=None 时取 current_player_id（decide 路径）；显式传入时用于 sweep 等没有当前行动者的场景。 """
    if me_id is None:
        me_id = llm_state["current_player_id"]
    me = next(p for p in llm_state["players"] if p["id"] == me_id)
    return (
        f"phase={llm_state['phase']} "
        f"hole={me['hole_cards']} "
        f"board={llm_state['community_cards']} "
        f"pot={llm_state['pot']} "
        f"to_call={llm_state['current_bet_this_round'] - me['current_bet']}"
    )


def topk_by_similarity(query_text, candidates, k, vec_lookup=None, salience_fn=None):
    if not candidates:
        return [], np.zeros(0)
    q = embed(query_text)[0]

    if vec_lookup is not None:
        # 跳过没存向量的 candidate（理论不会发生）
        vecs, kept = [], []
        for c in candidates:
            v = vec_lookup.get(c["id"])
            if v is None:
                continue
            vecs.append(v)
            kept.append(c)
        if not kept:
            return [], np.zeros(0)
        C = np.vstack(vecs).astype(np.float32)
        sims = C @ q
    else:
        # kept = candidates
        # C = embed([c["text"] for c in candidates])
        # sims = C @ q
        raise ValueError("vec_lookup is required; on-the-fly embed disabled")

    sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)

    if salience_fn is not None:
        weights = np.array([float(salience_fn(c["id"]) or 0.0) for c in kept], dtype=np.float32)
        sims = sims + np.log(np.maximum(weights, 1e-6))  # similarity + log(salience)

    if k >= len(kept):
        order = np.argsort(-sims)
    else:
        order = np.argsort(-sims)[:k]
    return [kept[i] for i in order], sims[order]
