from __future__ import annotations

from typing import Tuple

from .sparse2D import Sparse2D

Agent = int
Item = int
Bid = float
Triplet = tuple[Agent, Item, Bid]


class Bids:
    def __init__(self, agents_items_bids) -> None:
        self.sparse = Sparse2D(agents_items_bids)

    @property
    def agents(self) -> list[Agent]:
        return list(self.sparse.agent_ids)

    @property
    def items(self) -> list[Item]:
        return list(self.sparse.item_ids)

    @property
    def nnz(self) -> int:
        return self.sparse.data.nnz

    @property
    def shape(self) -> Tuple[int, int]:
        return self.sparse.data.shape

    def set(self, agent_id, item_id, bid) -> None:
        self.sparse.set(agent_id, item_id, bid)

    def get(self, agent_id, item_id) -> None | Bid:
        return self.sparse.get(agent_id, item_id)

    def remove(self, agent_id, item_id) -> None:
        self.sparse.remove(agent_id, item_id)

    def bids_from_agent_id(self, agent_id) -> list[Tuple[Agent, Item, None | Bid]]:
        return self.sparse.numbers_from_agent_id(agent_id)

    def bids_from_item_id(self, item_id) -> list[Tuple[Agent, Item, None | Bid]]:
        return self.sparse.numbers_from_item_id(item_id)

    def copy(self) -> Bids:
        return Bids(self.to_triplets())

    def remove_item(self, item_id) -> None:
        self.sparse.remove_item(item_id)

    def remove_agent(self, agent_id) -> None:
        self.sparse.remove_agent(agent_id)

    def to_triplets(self) -> list[Triplet]:
        return self.sparse.to_triplets()
