import random

from .cards import make_deck, evaluate_hand


PHASE_WAITING = "waiting"
PHASE_PREFLOP = "preflop"
PHASE_FLOP = "flop"
PHASE_TURN = "turn"
PHASE_RIVER = "river"
PHASE_SHOWDOWN = "showdown"
PHASE_HAND_OVER = "hand_over"


class GameController:
    def __init__(self, players):
        self.players = sorted(players, key=lambda p: p.seat)
        self.small_blind = 1
        self.big_blind = 2

        # 玩家查询表（玩家在一局内不变，建一次）
        self.players_by_seat = {p.seat: p for p in self.players}
        self.players_by_id = {p.player_id: p for p in self.players}
        self._seat_order = [p.seat for p in self.players]   # 升序

        # 牌局状态
        self.deck = []
        self.community_cards = []
        self.pot = 0
        self.current_bet_this_round = 0
        self.last_raise_size_this_round = self.big_blind
        self.dealer_seat = self._seat_order[0]
        self.phase = PHASE_WAITING
        self.current_player_seat = None
        self.has_acted_this_round = set()
        self.action_history = []
        self.winners = []
        self.showdown_ranks = {}    # pid -> hand_rank str，含输家；非摊牌手保持为空
        self.hand_finished = True
        self.hand_index = 0

    def start_hand(self):
        # --- 1) 重置本手状态 ---
        self.deck = make_deck()
        random.shuffle(self.deck)

        self.community_cards = []
        self.pot = 0
        self.current_bet_this_round = 0
        self.last_raise_size_this_round = self.big_blind
        self.has_acted_this_round = set()
        self.action_history = []
        self.winners = []
        self.showdown_ranks = {}
        self.hand_finished = False
        for p in self.players:
            p.busted = (p.stack == 0)
            p.folded = p.busted
            p.hole_cards = []
            p.current_bet = 0
            p.total_committed = 0

        # --- 2) 顺时针移动 dealer button（第一手不移动；跳过已 busted）---
        if self.hand_index > 0:
            n = len(self._seat_order)
            idx = self._seat_order.index(self.dealer_seat)
            for i in range(1, n + 1):
                cand = self._seat_order[(idx + i) % n]
                if not self.players_by_seat[cand].busted:
                    self.dealer_seat = cand
                    break

        self.phase = PHASE_PREFLOP
        self.hand_index += 1

        # --- 3) 算"本手参与顺序"：从 dealer 下家开始环绕，已 busted 跳过 ---
        n = len(self._seat_order)
        dealer_idx = self._seat_order.index(self.dealer_seat)
        hand_order = []
        for i in range(1, n + 1):  # 从 dealer+1 开始，dealer 本人在最后
            s = self._seat_order[(dealer_idx + i) % n]
            p = self.players_by_seat[s]
            if not p.busted:
                hand_order.append(p)

        if len(hand_order) < 2:
            # 不够人继续，直接收
            self.hand_finished = True
            self.phase = PHASE_HAND_OVER
            self.current_player_seat = None
            return

        # --- 4) 发底牌（每人 2 张，按 hand_order 轮两圈）---
        for _ in range(2):
            for p in hand_order:
                p.hole_cards.append(self.deck.pop())

        # --- 5) 下盲注 + 决定 preflop 第一个行动者 ---
        # heads-up（2 人）规则特殊：dealer = SB，且 SB 在 preflop 先行动
        if len(hand_order) == 2:
            bb_player = hand_order[0]                                # dealer 下家
            sb_player = self.players_by_seat[self.dealer_seat]       # dealer 本人
            first_to_act = sb_player
        else:
            sb_player = hand_order[0]
            bb_player = hand_order[1]
            first_to_act = hand_order[2]

        sb_amt = min(self.small_blind, sb_player.stack)
        bb_amt = min(self.big_blind, bb_player.stack)
        sb_player.stack -= sb_amt
        sb_player.current_bet = sb_amt
        sb_player.total_committed += sb_amt
        bb_player.stack -= bb_amt
        bb_player.current_bet = bb_amt
        bb_player.total_committed += bb_amt
        self.pot = sb_amt + bb_amt
        self.current_bet_this_round = bb_amt

        self.current_player_seat = first_to_act.seat
        if first_to_act.stack == 0 or first_to_act.folded or first_to_act.busted:
            self._advance_round_pointer(first_to_act.seat)

    def apply_action(self, action):
        """
        收到当前行动者的动作，推进游戏。
        action 形如 {"type": "fold" | "check" | "call" | "raise", "amount": int (仅 raise)}
        其中 raise 的 amount 是"加注到的总额"，不是差额。
        """
        seat = self.current_player_seat
        if seat is None:
            raise RuntimeError("当前没有行动者，无法 apply_action")
        p = self.players_by_seat[seat]

        # --- 0) 硬校验：必须落在 _legal_actions 内（controller 是真相源）---
        legal = self._legal_actions(p)
        action = self._validate_action(action, legal)
        atype = action["type"]

        # --- 1) 执行动作（更新筹码 / 底池 / 本街最高下注线）---
        is_effective_raise = None
        if atype == "fold":
            p.folded = True

        elif atype == "check":
            pass

        elif atype == "call":
            to_call = self.current_bet_this_round - p.current_bet
            amt = min(to_call, p.stack)            # 不够就 all-in for less
            p.stack -= amt
            p.current_bet += amt
            p.total_committed += amt
            self.pot += amt

        elif atype == "raise":
            prev_line = self.current_bet_this_round
            prev_last_raise = self.last_raise_size_this_round
            new_total = action["amount"]
            diff = new_total - p.current_bet
            diff = min(diff, p.stack)              # 超过自己 stack 则 all-in
            p.stack -= diff
            p.current_bet += diff
            p.total_committed += diff
            self.pot += diff
            is_effective_raise = False
            if p.current_bet > prev_line:
                raise_increment = p.current_bet - prev_line
                self.current_bet_this_round = p.current_bet
                if raise_increment >= prev_last_raise:
                    self.last_raise_size_this_round = raise_increment
                    self.has_acted_this_round = set()
                    is_effective_raise = True

        else:
            raise ValueError(f"未知动作类型: {atype}")

        self.has_acted_this_round.add(seat)
        entry = {
            "phase": self.phase,
            "player_id": p.player_id,
            "action": action,
        }
        if atype == "raise":
            # 短码 all-in 低于最小加注时 effective_raise=False，便于日志/诈唬率剔除
            entry["effective_raise"] = is_effective_raise
        self.action_history.append(entry)

        self._advance_round_pointer(seat)

    def _validate_action(self, action, legal):
        """ 严格校验上层传来的 action 是否在 _legal_actions 内；不合法直接抛。 """
        if not isinstance(action, dict) or "type" not in action:
            raise ValueError(f"非法动作（结构错误）：{action!r}")
        if not legal:
            raise RuntimeError(f"当前玩家无合法动作，但收到 {action!r}")
        atype = action["type"]
        legal_by_type = {a["type"]: a for a in legal}
        if atype not in legal_by_type:
            raise ValueError(f"动作 {atype} 不在合法列表 {sorted(legal_by_type)}")
        if atype == "raise":
            rule = legal_by_type[atype]
            amt = action.get("amount")
            if not isinstance(amt, int):
                raise ValueError(f"raise 缺少整数 amount：{action!r}")
            if amt < rule["min_amount"] or amt > rule["max_amount"]:
                raise ValueError(
                    f"raise amount={amt} 越界 [{rule['min_amount']}, {rule['max_amount']}]"
                )
            return {"type": "raise", "amount": amt}
        return {"type": atype}

    def _advance_round_pointer(self, prev_seat):
        """ 一个玩家行动完（或贴盲完）后，决定下一步：
        ① 只剩一人没弃牌 → 直接收底池；
        ② 本街已闭合 / 没人还能行动 → 推进到下一街或摊牌；
        ③ 否则把行动权交给 prev_seat 之后第一位还能行动的玩家。 """
        # ① 只剩一人没弃牌
        not_folded = [pp for pp in self.players if not pp.folded]
        if len(not_folded) == 1:
            winner = not_folded[0]
            winner.stack += self.pot
            self.winners = [{
                "player_id": winner.player_id,
                "amount": self.pot,
                "hand_rank": None,
            }]
            self.pot = 0
            self.phase = PHASE_HAND_OVER
            self.hand_finished = True
            self.current_player_seat = None
            return

        # ② 本街闭合：所有还能行动的人 current_bet 已对齐且都行动过
        can_act = [pp for pp in self.players if not pp.folded and pp.stack > 0]
        round_done = (
            all(pp.current_bet == self.current_bet_this_round for pp in can_act)
            and all(pp.seat in self.has_acted_this_round for pp in can_act)
        )
        if round_done:
            self._advance_street()
            return

        # ③ 找下一位能行动者（顺时针、跳过弃牌/all-in/busted）
        n = len(self._seat_order)
        idx = self._seat_order.index(prev_seat)
        for i in range(1, n + 1):
            s = self._seat_order[(idx + i) % n]
            pp = self.players_by_seat[s]
            if not pp.folded and not pp.busted and pp.stack > 0:
                self.current_player_seat = s
                return
        # 理论 ② 已接住，兜底
        self.current_player_seat = None
        self._advance_street()

    def get_state(self, viewer_id=None):
        """
        返回当前局面快照。
        viewer_id=None：上帝视角，所有 hole_cards 可见（用于日志）
        viewer_id=某玩家 id：只有该玩家自己 hole_cards 可见
        真正发生 showdown（showdown_ranks 非空）时：只公开未弃牌玩家的底牌；
        非摊牌结束（只剩一人收底池）不公开任何底牌。
        """
        god_view = viewer_id is None
        was_showdown = bool(self.showdown_ranks)

        players_out = []
        for p in self.players:
            visible = god_view or p.player_id == viewer_id or (was_showdown and not p.folded)
            players_out.append({
                "id": p.player_id,
                "seat": p.seat,
                "stack": p.stack,
                "current_bet": p.current_bet,
                "total_committed": p.total_committed,
                "folded": p.folded,
                "busted": p.busted,
                "hole_cards": [c.code for c in p.hole_cards] if visible else [],
            })

        current_p = self.players_by_seat.get(self.current_player_seat)
        legal = self._legal_actions(current_p) if current_p is not None else []

        return {
            "hand_index": self.hand_index,
            "phase": self.phase,
            "config": {"small_blind": self.small_blind, "big_blind": self.big_blind},
            "community_cards": [c.code for c in self.community_cards],
            "pot": self.pot,
            "current_bet_this_round": self.current_bet_this_round,
            "dealer_seat": self.dealer_seat,
            "players": players_out,
            "current_player_id": current_p.player_id if current_p is not None else None,
            "legal_actions": legal,
            "action_history": list(self.action_history),
            "winners": list(self.winners),
            "showdown_ranks": dict(self.showdown_ranks),
        }

    def _advance_street(self):
        """进入下一条街：发公共牌 → 清本街下注 → 找下一街首位。
        到 river 之后转 showdown。"""
        if self.phase == PHASE_PREFLOP:
            self.community_cards.extend(self.deck.pop() for _ in range(3))     # flop
            self.phase = PHASE_FLOP
        elif self.phase == PHASE_FLOP:
            self.community_cards.append(self.deck.pop())                       # turn
            self.phase = PHASE_TURN
        elif self.phase == PHASE_TURN:
            self.community_cards.append(self.deck.pop())                       # river
            self.phase = PHASE_RIVER
        elif self.phase == PHASE_RIVER:
            self.phase = PHASE_SHOWDOWN
            self._run_showdown()
            return

        # 清本街下注、找下一街第一个行动者（dealer 下家方向，第一个还能行动的）
        self.current_bet_this_round = 0
        self.last_raise_size_this_round = self.big_blind
        self.has_acted_this_round = set()
        for p in self.players:
            p.current_bet = 0

        n = len(self._seat_order)
        dealer_idx = self._seat_order.index(self.dealer_seat)
        first = None
        for i in range(1, n + 1):
            s = self._seat_order[(dealer_idx + i) % n]
            pp = self.players_by_seat[s]
            if not pp.folded and not pp.busted and pp.stack > 0:
                first = s
                break

        if first is None:
            # 剩下的都 all-in 了，没人能再下注：直接发完后面的牌进 showdown
            self.current_player_seat = None
            self._advance_street()
        else:
            self.current_player_seat = first

    def _run_showdown(self):
        """
        按分层（side pot）思路分底池：
        把每个 total_committed 值当成一个"层"。第 L 层的池子 = (L - 上一层L) * 至少投到 L 的人数；
        归属：在该层内还没弃牌的人中牌力最强者（用 evaluate_hand）。
        平手时按人均整除，余数从第一位开始 +1 分发。
        """
        levels = sorted({p.total_committed for p in self.players if p.total_committed > 0})

        # 给所有未弃牌玩家算牌力
        rankings = {}   # player_id -> (treys score, class_name)；score 越小越强
        for p in self.players:
            if not p.folded:
                rankings[p.player_id] = evaluate_hand(p.hole_cards, self.community_cards)

        # 留下所有摊牌玩家的牌型（输赢都要），供 game_logger 和诈唬率统计用
        self.showdown_ranks = {pid: rk[1] for pid, rk in rankings.items()}

        winners_log = []
        prev = 0
        for level in levels:
            layer_size = level - prev
            contributors = [p for p in self.players if p.total_committed >= level]
            layer_pot = layer_size * len(contributors)
            eligible = [p for p in contributors if not p.folded]

            if not eligible:
                # 该层没有未弃牌的争夺者（理论很难触发）：按贡献退回
                share = layer_pot // len(contributors)
                for pp in contributors:
                    pp.stack += share
                prev = level
                continue

            best_score = min(rankings[pp.player_id][0] for pp in eligible)
            winners_here = [pp for pp in eligible if rankings[pp.player_id][0] == best_score]
            share = layer_pot // len(winners_here)
            remainder = layer_pot - share * len(winners_here)
            for i, pp in enumerate(winners_here):
                extra = 1 if i < remainder else 0
                pp.stack += share + extra
                winners_log.append({
                    "player_id": pp.player_id,
                    "amount": share + extra,
                    "hand_rank": rankings[pp.player_id][1],
                    "layer_level": level,
                })
            prev = level

        self.winners = winners_log
        self.pot = 0
        self.phase = PHASE_HAND_OVER
        self.hand_finished = True
        self.current_player_seat = None

    def _legal_actions(self, p):
        """给当前行动者列出合法动作。raise 的 min_amount / max_amount 都是"加注到的总额"。"""
        if p is None or p.folded or p.busted or p.stack == 0:
            return []
        if self.phase in (PHASE_SHOWDOWN, PHASE_HAND_OVER, PHASE_WAITING):
            return []
        if self.current_player_seat != p.seat:
            return []

        actions = [{"type": "fold"}]
        to_call = self.current_bet_this_round - p.current_bet
        if to_call == 0:
            actions.append({"type": "check"})
        else:
            actions.append({"type": "call"})

        # raise 需要 call 完之后还能继续加；
        # 但若本玩家本街已行动过、现在只是面对一次"短码 all-in（reopens=False）"的补差额，
        # 行动集合并未重开，不应再给 raise（只能 fold/call）。
        already_acted_facing_short_allin = (
            to_call > 0 and p.seat in self.has_acted_this_round
        )
        if p.stack > to_call and not already_acted_facing_short_allin:
            # 最小再加注 = 当前线 + 上一次有效加注额（每条街起始为大盲）
            min_to = self.current_bet_this_round + self.last_raise_size_this_round
            max_to = p.current_bet + p.stack                        # all-in 上限
            if max_to >= min_to:
                actions.append({"type": "raise", "min_amount": min_to, "max_amount": max_to})
            else:
                # 短码 all-in：金额低于正常 min_to，仍允许出牌，但不会重开行动集合
                actions.append({
                    "type": "raise",
                    "min_amount": max_to,
                    "max_amount": max_to,
                    "reopens": False,
                })

        return actions
