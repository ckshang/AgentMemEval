import json
import shutil

from pathlib import Path

from src.game.controller import GameController
from src.game.game_logger import write_hand_rows
from .agent_players.fact import FactAgent
from .agent_players.expr import ExprAgent
from .agent_players.fxsync import FactExprSyncAgent
from .agent_players.fxasync import FactExprAsyncAgent
from .agent_players.naive import NaiveLLMPlayer
from .agent_players.mbti import MBTIAgent
from .personas import MBTI_TYPES


_ALGO_CLS = {
    "fact": FactAgent,
    "expr": ExprAgent,
    "fxsync": FactExprSyncAgent,
    "fxasync": FactExprAsyncAgent,
}


_MEMORY_FILES = [
    "facts.jsonl",
    "fact_embeddings.npy",
    "facts_state.jsonl",
    "experience.md",
    "experience_log.jsonl",
    "trajectory_log.jsonl",
    "actions.jsonl",
    "sweep_log.jsonl",
]


# 泛化测试时只复活"长期记忆"——decide 时会真正用到的部分。
# trajectory_log / actions / sweep_log / experience_log 是工作日志/历史，
# 复制过去会污染评估期的新日志（append 模式），且会让冻结 agent 的
# trajectory 同时混入训练桌和评估桌的 hand_index。
_REVIVE_FILES = [
    "facts.jsonl",
    "fact_embeddings.npy",
    "facts_state.jsonl",
    "experience.md",
]


def snapshot_player_memory(player, snapshot_dir):
    """把 player.output_dir 下所有记忆文件拷到 snapshot_dir。
    新 agent 没有 memory 文件（比如 NaiveLLM）则跳过。"""
    src = Path(player.output_dir)
    dst = Path(snapshot_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for fname in _MEMORY_FILES:
        s = src / fname
        if s.exists():
            shutil.copy2(s, dst / fname)


def algo_of_player(player):
    pid = player.player_id
    parts = pid.split("_", 2)
    if len(parts) < 3:
        return None
    tail = parts[2]
    algo = tail.split("-")[0]
    if algo in _ALGO_CLS or algo in MBTI_TYPES:
        return algo
    return None


def build_agent_from_snapshot(algo, snapshot_dir, player_id, starting_stack, model_name,
                              new_output_dir):
    """
    从 snapshot 复活一个 agent：
    1) 在 new_output_dir 下复制一份 snapshot 内容（保持文件独立，不污染原快照）
    2) 用对应 cls 初始化，构造函数会自动 load 这些文件
    """
    new_dir = Path(new_output_dir)
    new_dir.mkdir(parents=True, exist_ok=True)
    for fname in _REVIVE_FILES:
        s = Path(snapshot_dir) / fname
        if s.exists():
            shutil.copy2(s, new_dir / fname)
    if algo in MBTI_TYPES:
        agent = MBTIAgent(
            player_id=player_id,
            model_name=model_name,
            starting_stack=starting_stack,
            output_dir=str(new_dir),
            persona=algo,
        )
    else:
        agent = _ALGO_CLS[algo](
            player_id=player_id,
            model_name=model_name,
            starting_stack=starting_stack,
            output_dir=str(new_dir),
        )
    agent.frozen = True
    return agent


def run_generalization_table(snapshot_dir, algo, model_name, *,
                             player_id_in_snap,
                             new_output_dir,
                             n_opponents=5,
                             hands=10,
                             starting_stack=1000,
                             naive_model=None):
    """
    把一个 snapshot 复活成 seat 0 的玩家，与 n_opponents 个 NaiveLLM 同桌打 hands 手。
    返回 list[dict]，每手一行：{"hand_index": h, "stacks": {pid: stack, ...}}。
    """
    naive_model = naive_model or model_name
    new_output_dir = Path(new_output_dir)
    new_output_dir.mkdir(parents=True, exist_ok=True)

    pid_main = f"player_00_{algo}-snap"
    main_dir = new_output_dir / pid_main
    main_player = build_agent_from_snapshot(
        algo=algo,
        snapshot_dir=snapshot_dir,
        player_id=pid_main,
        starting_stack=starting_stack,
        model_name=model_name,
        new_output_dir=str(main_dir),
    )

    players = [main_player]
    for seat in range(1, n_opponents + 1):
        pid = f"player_{seat:02d}_naive"
        naive_dir = new_output_dir / pid
        players.append(NaiveLLMPlayer(player_id=pid, model_name=naive_model, starting_stack=starting_stack, output_dir=str(naive_dir)))

    controller = GameController(players=players)

    rows = []
    for _ in range(hands):
        if sum(1 for p in players if p.stack > 0) < 2:
            break
        controller.start_hand()
        while not controller.hand_finished:
            seat = controller.current_player_seat
            if seat is None:
                break
            cur = controller.players_by_seat[seat]
            st = controller.get_state(viewer_id=cur.player_id)
            action = cur.decide(st)
            controller.apply_action(action)

        final_state = controller.get_state(viewer_id=None)
        for p in players:
            p.observe(final_state)
        write_hand_rows(str(new_output_dir), controller, final_state)

        rows.append({
            "hand_index": controller.hand_index,
            "stacks":     {p.player_id: int(p.stack) for p in players},
            "main_pid":   pid_main,
        })

    # 落盘原始记录供事后核查
    with open(new_output_dir / "generalization_log.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return rows
