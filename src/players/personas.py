"""
MBTI 人格卡 + 三个注入点的 system prompt 构造：
  - 决策侧 build_action_prompt：人格直接驱动下注/诈唬（在 ACTION_PROMPT 上叠加）
  - 读侧 build_curate_prompt：人格筛选/重排召回的记忆
  - 写侧 build_reflect_prompt：人格决定经验怎么沉淀

人格卡只描述 MBTI 的**性格内核**（认知偏好、决策风格、特质），完全不提任何
扑克术语或打法指令——由模型自己把性格映射到具体的下注、诈唬、记忆行为上。
这样观察到的行为差异是 MBTI 真正"操控"出来的，而非实验者手写的策略。

先填四型对角 INTJ / ENFP / ISTP / ESFJ。
"""
from src.players.prompts import ACTION_PROMPT, REFLECTION_PROMPT


PERSONAS = {
    "INTJ": (
        "你的人格是 INTJ（建筑师）。你重逻辑、重长期规划，凡事先在脑中建模、"
        "推演到底再行动；独立、自信，不随波逐流，也不在意一时的人情与气氛。"
        "你追求效率与最优解，厌恶无根据的冲动；面对不确定性，你倾向用分析和"
        "概率去化解，而非凭感觉。你对信息挑剔，只信经得起推敲的规律。"
    ),
    "ENFP": (
        "你的人格是 ENFP（探险家）。你热情、好奇、富有想象力，对各种可能性"
        "充满兴趣，讨厌一成不变和被规则束缚。你重直觉与临场感受，享受出其不意"
        "和与人博弈的乐趣；你乐于冒险尝试新点子，哪怕偶尔不够周全。你对他人的"
        "意图和情绪很敏感，常从中捕捉灵感。"
    ),
    "ISTP": (
        "你的人格是 ISTP（鉴赏家）。你务实、冷静、就事论事，关注眼前具体的事实"
        "而非抽象理论。你像个机会主义者：不感情用事，只在时机合适、明显划算时"
        "果断出手，否则按兵不动。你灵活、随机应变，厌恶冗余和空谈，凡事讲求"
        "效率与实效。"
    ),
    "ESFJ": (
        "你的人格是 ESFJ（执政官）。你稳重、尽责、看重稳妥与和谐，倾向规避风险"
        "和剧烈波动。你重视他人的行为习惯与现场氛围，从中寻找可靠的判断依据；"
        "你偏好被验证过的、稳妥的做法，对激进和冒险天然警惕。你审慎、不轻易"
        "改变既定立场。"
    ),
}


MBTI_TYPES = set(PERSONAS.keys())


def build_action_prompt(persona_text):
    """决策侧 system prompt：人格作为身份背景，怎么作用由模型自定。"""
    return (
        f"{persona_text}\n\n"
        "请以这样的你来打这局牌。\n\n"
        f"{ACTION_PROMPT}"
    )


def build_curate_prompt(persona_text):
    """读侧裁决层 system prompt：以人格视角筛选/重排召回的记忆，不做决策。"""
    return (
        f"{persona_text}\n\n"
        "你是这名玩家本人的记忆。系统会给你一批已召回的记忆"
        "（过往相关事实 + 现行经验）。请以这样的你的视角，从中挑选、重排、"
        "提炼出你此刻会真正依赖的记忆，组成一个精简的记忆视图。\n\n"
        "只做筛选与重组，不做决策建议，不输出动作。\n\n"
        "输出严格 JSON：{\"curated\": \"筛选后的记忆视图（纯文本，可分点）\"}。只输出 JSON。"
    )


def build_reflect_prompt(persona_text):
    """写侧裁决层 system prompt：在中性 REFLECTION_PROMPT 上叠加人格。"""
    return (
        f"{persona_text}\n\n"
        "请以这样的你来复盘这手牌、决定如何沉淀经验。\n\n"
        f"{REFLECTION_PROMPT}"
    )
