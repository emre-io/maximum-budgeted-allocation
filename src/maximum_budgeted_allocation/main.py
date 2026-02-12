import csv

from .data.bids import Bids
from .data.budgets import Budgets
from .data.instance import Instance
from .solvers.approximationalgorithm import solve_apx

bids_as_triplets = []
with open("./data/example_3/bids.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for agent_id, item_id, bid in reader:
        bids_as_triplets.append((int(agent_id), int(item_id), float(bid)))

budgets_as_tuples = []
with open("./data/example_3/budgets.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for agent_id, budget in reader:
        budgets_as_tuples.append((int(agent_id), float(budget)))

bids = Bids(bids_as_triplets)
budgets = Budgets(budgets_as_tuples)
instance = Instance(bids, budgets)

allocation, revenue, lp_revenue = solve_apx(instance)

print(instance.is_feasible())
print(allocation.is_feasible(budgets))
# print(allocation.to_triplets())
print(f"Allocation achieves {revenue/lp_revenue * 100} percent of LP optimum.")
