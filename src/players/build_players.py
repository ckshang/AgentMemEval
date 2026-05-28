import os

from .agent_players.fact import FactAgent
from .agent_players.expr import ExprAgent
from .agent_players.fxsync import FactExprSyncAgent
from .agent_players.fxasync import FactExprAsyncAgent
from .agent_players.naive import NaiveLLMPlayer


_AGENT_ALGO = {
    "fact": FactAgent,
    "expr": ExprAgent,
    "fxsync": FactExprSyncAgent,
    "fxasync": FactExprAsyncAgent,
    "naive": NaiveLLMPlayer,
}


def build_players(player_names, starting_stack, output_dir):
    """
    构建玩家。

    player_name 名字规则:
      - "{algo}-{model}" → e.g., "fact-deepseekv4f"
    """
    if len(player_names) < 2 or len(player_names) > 12:
        raise ValueError("Players must be between 2 and 12")

    players = []
    for seat, name in enumerate(player_names):
        player_id = f"player_{seat:02d}_{name}"
        player_dir = os.path.join(output_dir, player_id)

        algo_name, model_name = name.split("-")
        if algo_name not in _AGENT_ALGO.keys():
            raise ValueError(f"{algo_name} is not supported yet")

        player = _AGENT_ALGO[algo_name](
            player_id=player_id,
            model_name=model_name,
            starting_stack=starting_stack,
            output_dir=player_dir,
        )
        players.append(player)

    return players