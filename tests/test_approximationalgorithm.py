import csv

import pytest

from maximum_budgeted_allocation import (
    Bids,
    Budgets,
    Instance,
    solve_apx,
)


def load_instance(example_name: str) -> Instance:
    bids_as_triplets = []
    with open(f"./data/{example_name}/bids.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for agent_id, item_id, bid in reader:
            bids_as_triplets.append((int(agent_id), int(item_id), float(bid)))

    budgets_as_tuples = []
    with open(f"./data/{example_name}/budgets.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for agent_id, budget in reader:
            budgets_as_tuples.append((int(agent_id), float(budget)))

    bids = Bids(bids_as_triplets)
    budgets = Budgets(budgets_as_tuples)
    return Instance(bids, budgets)


@pytest.mark.parametrize(
    "example_name",
    ["example_1", "example_2", "example_3"],
)
def test_instance_is_feasible(example_name):
    instance = load_instance(example_name)
    assert instance.is_feasible()


@pytest.mark.parametrize(
    "example_name",
    ["example_1", "example_2", "example_3"],
)
def test_apx_solution_is_feasible(example_name):
    instance = load_instance(example_name)
    allocation, revenue, lp_revenue = solve_apx(instance)

    assert allocation is not None
    assert revenue >= 0
    assert lp_revenue >= 0
    assert revenue >= 0.75 * lp_revenue
    assert allocation.is_feasible(instance.budgets)
