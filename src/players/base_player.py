class BasePlayer:
    """
    玩家基类。
    对外只有 decide / observe 两个函数。
    """
    def __init__(self, player_id, model_name, starting_stack, output_dir):
        self.player_id = player_id
        self.seat = int(player_id.split("_")[1])
        self.model_name = model_name
        self.stack = starting_stack
        self.output_dir = output_dir

        self.hole_cards = []
        self.current_bet = 0
        self.total_committed = 0
        self.folded = False
        self.busted = False
        self.frozen = False

    def decide(self, state):
        """ 用于做决策。 """
        raise NotImplementedError

    def observe(self, final_state):
        """ 用于反思并记忆。"""
        pass
