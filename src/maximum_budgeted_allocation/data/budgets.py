from __future__ import annotations

Agent = int


class Budgets:
    def __init__(self, agent_budget) -> None:
        self.data = {}
        if agent_budget is not None:
            for agent, budget in agent_budget:
                self.set(agent, budget)

    def set(self, agent, budget) -> None:
        if budget < 0.0:
            raise ValueError("Budget has to be positive, got " + str(budget) + ".")
        self.data[agent] = budget

    def get(self, agent) -> float:
        return self.data.get(agent, 0.0)

    def remove(self, agent) -> None:
        self.data.pop(agent, None)

    def copy(self) -> Budgets:
        b = Budgets(None)
        b.data = self.data.copy()
        return b
