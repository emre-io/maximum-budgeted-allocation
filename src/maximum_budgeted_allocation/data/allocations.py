from __future__ import annotations

from typing import List, Set, Tuple

from .sparse2D import Sparse2D

Agent = int
Item = int
Items = Set[int]
Payment = float


class Allocations:
    def __init__(self, agents_items_payments) -> None:
        self.sparse = Sparse2D(agents_items_payments)

    @property
    def agents(self) -> List[Agent]:
        return list(self.sparse.agent_ids)

    @property
    def items(self) -> List[Item]:
        return list(self.sparse.item_ids)

    @property
    def nnz(self) -> int:
        return self.sparse.data.nnz

    @property
    def shape(self) -> Tuple[int, int]:
        return self.sparse.data.shape

    def add(self, agent_id, item_id, bid) -> None:
        self.sparse = self.sparse.add(agent_id, item_id, bid)

    def set(self, agent_id, item_id, bid) -> None:
        self.sparse.set(agent_id, item_id, bid)

    def get(self, agent_id, item_id) -> None | Payment:
        return self.sparse.get(agent_id, item_id)

    def remove(self, agent_id, item_id) -> None:
        self.sparse.remove(agent_id, item_id)

    def items_of_agent(self, agent_id) -> Items:
        return self.sparse.agent_ids_to_item_ids[agent_id]

    def payments_from_agent_id(self, agent_id) -> List[Tuple[Agent, Item, None | float]]:
        return self.sparse.numbers_from_agent_id(agent_id)

    def payments_from_item_id(self, item_id) -> List[Tuple[Agent, Item, None | float]]:
        return self.sparse.numbers_from_item_id(item_id)

    def total_revenue(self) -> float:
        payments = [payment for (_, _, payment) in self.to_triplets()]
        return sum(payments)

    def to_triplets(self) -> List[Tuple[Agent, Item, float]]:
        return self.sparse.to_triplets()

    def is_feasible(self, budgets) -> bool:
        for agent_id in self.agents:
            payments_from_agent: List[None | float] = [
                payment for (_, _, payment) in self.payments_from_agent_id(agent_id)
            ]
            if sum(payments_from_agent) > budgets.get(agent_id):  # ty:ignore[no-matching-overload]
                raise ValueError(
                    f"Agent {agent_id} can not pay for all assigned items. "
                    f"His payment is {sum(payments_from_agent)} and his budget is {budgets.get(agent_id)}."  # ty:ignore[no-matching-overload]
                )

        for item_id in self.items:
            allocations = [payment for (_, _, payment) in self.payments_from_item_id(item_id)]

            if len(allocations) > 1:
                raise ValueError(
                    f"Item {item_id} is allocated more than once "
                    f"({len(allocations)} allocations)."
                )

        return True
