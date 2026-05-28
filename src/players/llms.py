import os
import re
import json
import openai


_PROVIDERS = {
    "deepseekv4f": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model_id": "deepseek-v4-flash",
    },
    "deepseekv4p": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model_id": "deepseek-v4-pro",
    },
    "deepseekr1": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model_id": "deepseek-reasoner",
    },
}


_clients = {}


def call_llm(model_name, messages, temperature=1.0, max_tokens=2048, json_mode=False):
    if model_name not in _PROVIDERS:
        raise ValueError(f"{model_name} is not supported yet")

    if model_name not in _clients:
        cfg = _PROVIDERS[model_name]
        api_key = os.getenv(cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(f"No API key found for {model_name}")
        _clients[model_name] = openai.OpenAI(api_key=api_key, base_url=cfg["base_url"])

    kwargs = {
        "model": _PROVIDERS[model_name]["model_id"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return _clients[model_name].chat.completions.create(**kwargs).choices[0].message.content


def _try_load_json(text):
    if not text:
        return None
    # 优先匹配最外层 {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _normalize_action(action, legal_actions):
    """ 校验 LLM 给的 action 是否合法；不合法返回 None。 """
    if not isinstance(action, dict) or "type" not in action:
        return None
    atype = action["type"]
    legal_by_type = {a["type"]: a for a in legal_actions}
    if atype not in legal_by_type:
        return None
    if atype == "raise":
        rule = legal_by_type[atype]
        amount = action.get("amount")
        if not isinstance(amount, int):
            return None
        # 落到合法区间
        amount = max(rule["min_amount"], min(amount, rule["max_amount"]))
        return {"type": atype, "amount": amount}
    return {"type": atype}


def _safe_fallback(legal_actions):
    """ LLM 输出无法解析时的兜底：优先 check，否则 fold，否则第一个合法动作。 """
    by_type = {a["type"]: a for a in legal_actions}
    for pref in ("check", "fold", "call"):
        if pref in by_type:
            return {"type": pref}
    return {"type": legal_actions[0]["type"]} if legal_actions else {"type": "fold"}


def parse_response(text, legal_actions):
    parsed = _try_load_json(text)

    action = None
    if parsed and isinstance(parsed.get("action"), dict):
        action = _normalize_action(parsed["action"], legal_actions)

    if action is None:
        action = _safe_fallback(legal_actions)
    return parsed, action
