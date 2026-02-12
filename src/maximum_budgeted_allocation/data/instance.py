from .bids import Bids
from .budgets import Budgets


class Instance:
    def __init__(self, bids, budgets):
        self.bids: Bids = bids
        self.budgets: Budgets = budgets

        pass

    def is_feasible(self):
        for agent in self.bids.agents:
            max_bid_agent = 0.0
            for _, _, bid in self.bids.bids_from_agent_id(agent):
                if bid > max_bid_agent:
                    max_bid_agent = bid

            budget_agent = self.budgets.get(agent)
            if max_bid_agent > budget_agent:
                raise ValueError(
                    f"Instance is not feasbible. "
                    f"Agent {agent} max bid {max_bid_agent} is greater than "
                    f"his budget {budget_agent}."
                )

        return True
