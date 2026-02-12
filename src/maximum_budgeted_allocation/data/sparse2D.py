from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from scipy.sparse import dok_matrix

Agent = int
Agent_Index = int
Item = int
Item_Index = int
Triplet = tuple[Agent, Item, float]


class Sparse2D:
    def __init__(self, agents_items_number) -> None:

        self.agent_ids: Set[Agent] = {agent for (agent, _, _) in agents_items_number}
        self.item_ids: Set[Agent] = {item for (_, item, _) in agents_items_number}

        self.agent_indices_to_ids: Dict[Agent_Index, Agent] = {
            agent_index: agent_id for agent_index, agent_id in enumerate(self.agent_ids)
        }
        self.item_indices_to_ids: Dict[Item_Index, Item] = {
            item_index: item_id for item_index, item_id in enumerate(self.item_ids)
        }

        self.agent_ids_to_indices: dict[Agent, Agent_Index] = {
            agent_id: agent_idx for agent_idx, agent_id in self.agent_indices_to_ids.items()
        }
        self.item_ids_to_indices: dict[Item, Item_Index] = {
            item_id: item_idx for item_idx, item_id in self.item_indices_to_ids.items()
        }

        self.agent_ids_to_item_ids: Dict[Agent, Set[Item]] = defaultdict(set)
        self.item_ids_to_agent_ids: Dict[Item, Set[Agent]] = defaultdict(set)

        self.num_agents: int = len(self.agent_ids)
        self.num_items: int = len(self.item_ids)
        self.data: dok_matrix = dok_matrix((self.num_agents, self.num_items), dtype=float)

        for agent_id, item_id, number in agents_items_number:
            agent_index = self.agent_ids_to_indices[agent_id]
            item_index = self.item_ids_to_indices[item_id]

            self.data[agent_index, item_index] = number

            self.agent_ids_to_item_ids[agent_id].add(item_id)
            self.item_ids_to_agent_ids[item_id].add(agent_id)

    @property
    def agents(self) -> List[Agent]:
        return list(self.agent_ids)

    @property
    def items(self) -> List[Item]:
        return list(self.item_ids)

    @property
    def agents_to_items(self) -> Dict[Agent, Set[Item]]:
        return self.agent_ids_to_item_ids

    @property
    def items_to_agents(self) -> Dict[Item, Set[Agent]]:
        return self.item_ids_to_agent_ids

    @property
    def nnz(self) -> int:
        return self.data.nnz

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    # TODO: make slow adding fast.
    def add(self, agent_id, item_id, number) -> Sparse2D:
        triplets: List[Triplet] = self.to_triplets()
        triplets.append((agent_id, item_id, number))
        return Sparse2D(triplets)

    def set(self, agent_id, item_id, number) -> None:
        if agent_id not in self.agent_ids or item_id not in self.item_ids:
            return None

        agent_index = self.agent_ids_to_indices[agent_id]
        item_index = self.item_ids_to_indices[item_id]

        self.data[agent_index, item_index] = number

    def get(self, agent_id, item_id) -> None | float:
        if agent_id not in self.agent_ids or item_id not in self.item_ids:
            return None

        agent_index = self.agent_ids_to_indices[agent_id]
        item_index = self.item_ids_to_indices[item_id]

        return self.data[agent_index, item_index]

    def remove(self, agent_id, item_id) -> Tuple[Agent, Item] | None:
        if agent_id not in self.agent_ids or item_id not in self.item_ids:
            return None

        agent_index = self.agent_ids_to_indices[agent_id]
        item_index = self.item_ids_to_indices[item_id]
        key = (agent_index, item_index)
        self.data.pop(key)

        update_agents = False
        update_items = False

        if len(self.numbers_from_item_id(item_id)) == 0:
            update_items = True

        if len(self.numbers_from_agent_id(agent_id)) == 0:
            update_agents = True

        if update_agents:
            self.agent_ids_to_indices.pop(agent_id)
            self.agent_indices_to_ids.pop(agent_index)
            self.agent_ids.remove(agent_id)

            self.agent_ids_to_item_ids.pop(agent_id)

        for item, agents in self.item_ids_to_agent_ids.items():
            if agent_id in agents and item == item_id:
                agents.remove(agent_id)

        if update_items:
            self.item_ids_to_indices.pop(item_id)
            self.item_indices_to_ids.pop(item_index)
            self.item_ids.remove(item_id)

            self.item_ids_to_agent_ids.pop(item_id)

        for agent, items in self.agent_ids_to_item_ids.items():
            if item_id in items and agent == agent_id:
                items.remove(item_id)

        return (agent_id, item_id)

    def numbers_from_agent_id(self, agent_id: Agent) -> list[Tuple[Agent, Item, None | float]]:
        if agent_id not in self.agent_ids:
            return []

        item_ids: Set[Item] = self.agent_ids_to_item_ids[agent_id]

        result: list[Tuple[Agent, Item, None | float]] = []
        for item_id in item_ids:
            result.append((agent_id, item_id, self.get(agent_id, item_id)))

        return result

    def numbers_from_item_id(self, item_id: Item) -> list[Tuple[Agent, Item, None | float]]:
        if item_id not in self.item_ids:
            return []

        agent_ids: Set[Agent] = self.item_ids_to_agent_ids[item_id]

        result: list[Tuple[Agent, Item, None | float]] = []
        for agent_id in agent_ids:
            result.append((agent_id, item_id, self.get(agent_id, item_id)))

        return result

    def copy(self) -> Sparse2D:
        return Sparse2D(self.to_triplets())

    def remove_item(self, item_id) -> List[Tuple[Agent, Item] | None] | None:
        if item_id not in self.item_ids:
            return None

        agents_to_remove = self.numbers_from_item_id(item_id)
        return [self.remove(agent_id, item_id) for (agent_id, item_id, _) in agents_to_remove]

    def remove_agent(self, agent_id) -> List[Tuple[Agent, Item] | None] | None:
        if agent_id not in self.agent_ids:
            return None

        items_to_remove = self.numbers_from_agent_id(agent_id)
        return [self.remove(agent_id, item_id) for (agent_id, item_id, _) in items_to_remove]

    def to_triplets(self) -> List[Tuple[Agent, Item, float]]:
        return [
            (self.agent_indices_to_ids[agent_idx], self.item_indices_to_ids[item_idx], number)
            for (agent_idx, item_idx), number in self.data.items()
        ]
