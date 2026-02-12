from typing import Dict, Tuple

from ortools.linear_solver.pywraplp import Solver, Variable

from ..data.bids import Bids
from ..data.budgets import Budgets
from ..data.instance import Instance

Agent = int
Item = int
Allocation = float
Revenue = float


def solve_max_lp(
    bids: Bids,
    budgets: Budgets,
    solver_name="GLOP",
) -> Tuple[Dict[Tuple[Agent, Item], Allocation], Revenue]:

    solver = Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"Solver {solver_name} can not be initialized.")

    agent_ids = bids.agents
    item_ids = bids.items
    x: Dict[Tuple[Agent, Item], Variable] = {}

    for agent_id in agent_ids:
        for _, item_id, _ in bids.bids_from_agent_id(agent_id):
            x[agent_id, item_id] = solver.NumVar(0.0, 1.0, f"x[{agent_id},{item_id}]")

    # Objective function
    total_revenues_from_agents = [
        bids.get(agent_id, item_id) * x[agent_id, item_id] for agent_id, item_id in x.keys()
    ]

    solver.Maximize(solver.Sum(total_revenues_from_agents))

    # Budgetrestrictions
    for agent_id in agent_ids:
        payments_from_agent = [
            bids.get(agent_id, item_id) * x[agent_id, item_id]
            for _, item_id, _ in bids.bids_from_agent_id(agent_id)
        ]
        if payments_from_agent:
            solver.Add(solver.Sum(payments_from_agent) <= budgets.get(agent_id))

    # Itemrestrictions
    for item_id in item_ids:
        times_j_assigned = [x[a, item_id] for a, _, _ in bids.bids_from_item_id(item_id)]
        if times_j_assigned:
            solver.Add(solver.Sum(times_j_assigned) <= 1.0)

    status: int = solver.Solve()
    if status != Solver.OPTIMAL:
        raise RuntimeError(f"Optimal solution not found, status={status}")

    x_floats = {
        (agent_id, item_id): var.solution_value()
        for (agent_id, item_id), var in x.items()
        if var.solution_value() > 1e-12
    }

    objective = solver.Objective().Value()
    return x_floats, objective


def solve_min_lp(
    bids: Bids,
    budgets: Budgets,
    optimal_revenue: float,
    solver_name="GLOP",
) -> Dict[Tuple[Agent, Item], Allocation]:

    solver = Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"Solver {solver_name} can not be initialized.")

    agent_ids = bids.agents
    item_ids = bids.items
    x: Dict[Tuple[Agent, Item], Variable] = {}

    for a in agent_ids:
        for _, j, _ in bids.bids_from_agent_id(a):
            x[a, j] = solver.NumVar(0.0, 1.0, f"x[{a},{j}]")

    # Optimality restriction from max lp.
    total_revenues_from_agents = [bids.get(a, j) * x[a, j] for a, j in x.keys()]
    solver.Add(solver.Sum(total_revenues_from_agents) == optimal_revenue)

    # Budgetrestrictions
    for agent_id in agent_ids:
        payments_from_agent = [
            bids.get(agent_id, item_id) * x[agent_id, item_id]
            for _, item_id, _ in bids.bids_from_agent_id(agent_id)
        ]
        if payments_from_agent:
            solver.Add(solver.Sum(payments_from_agent) <= budgets.get(agent_id))

    # Itemrestrictions
    for item_id in item_ids:
        times_j_assigned = [x[a, item_id] for a, _, _ in bids.bids_from_item_id(item_id)]
        if times_j_assigned:
            solver.Add(solver.Sum(times_j_assigned) <= 1.0)

    # Objective function
    solver.Minimize(solver.Sum(x[a, j] for a, j in x.keys()))

    status: int = solver.Solve()
    if status != Solver.OPTIMAL:
        raise RuntimeError(f"Optimal solution not found, status={status}")

    x_floats = {
        (agent_id, item_id): var.solution_value()
        for (agent_id, item_id), var in x.items()
        if var.solution_value() > 1e-12
    }

    return x_floats


def solve_lp(instance: Instance) -> Tuple[Dict[Tuple[Agent, Item], Allocation], Revenue]:
    _, optimal_revenue = solve_max_lp(instance.bids, instance.budgets)
    x = solve_min_lp(instance.bids, instance.budgets, optimal_revenue)
    return x, optimal_revenue
