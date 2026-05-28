from enum import Enum

from treys import Card as TCard, Evaluator


class Suit(str, Enum):
    """花色"""
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"


class Rank(str, Enum):
    """牌序"""
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"  # 避免两位数有冲突
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


RANK_ORDER = [
    Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
    Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
    Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
]
SUIT_ORDER = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    @property
    def code(self):
        return f"{self.rank.value}{self.suit.value}"

    def __repr__(self):
        return f"Card({self.code})"


def make_deck():
    """牌堆（52张牌）"""
    return [Card(s, r) for s in SUIT_ORDER for r in RANK_ORDER]


_evaluator = Evaluator()


def evaluate_hand(hole_cards, community_cards):
    """评估"""
    hand = [TCard.new(c.code) for c in hole_cards]
    board = [TCard.new(c.code) for c in community_cards]
    score = _evaluator.evaluate(board, hand)
    class_id = _evaluator.get_rank_class(score)
    return score, _evaluator.class_to_string(class_id)
