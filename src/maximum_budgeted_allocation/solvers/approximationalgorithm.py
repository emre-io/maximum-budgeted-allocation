from typing import Tuple

from ..data.allocations import Allocations
from ..data.graph import (
    create_graph_from_instance,
    find_agent_with_only_leaf_items,
    find_tight_leaf_agent,
    find_virtual_agent_with_x_eq_one,
)
from ..data.instance import Instance
from .lp_solver import solve_lp

Revenue = float


def solve_apx(
    instance, x_min: float = 1e-12, tight_tol: float = 1e-12
) -> Tuple[Allocations, Revenue, float]:

    def _assign_and_delete_item(agent_id, item_id):

        agent_spent = sum(
            [payment for _, _, payment in allocation.payments_from_agent_id(agent_id)]
        )

        payment_for_item = min(
            instance.bids.get(agent_id, item_id),
            max(0.0, instance.budgets.get(agent_id) - agent_spent),
        )

        allocation.add(agent_id, item_id, payment_for_item)
        instance_copy.bids.remove_item(item_id)

    instance_copy = Instance(instance.bids.copy(), instance.budgets.copy())
    virtual_agents = {agent: False for agent in instance_copy.bids.agents}
    allocation = Allocations(())

    apx_factor = 0.75

    step = 0
    start_lp_opt = None

    while True:
        x, lp_opt = solve_lp(instance_copy)
        if start_lp_opt is None:
            start_lp_opt = lp_opt

        support = x.keys()
        # remove bids which are not in the support.
        for agent in instance_copy.bids.agents:
            for _, item, _ in instance_copy.bids.bids_from_agent_id(agent):
                if (agent, item) not in support:
                    instance_copy.bids.remove(agent, item)

        graph = create_graph_from_instance(instance_copy, x, x_min, virtual_agents, tight_tol)

        # No more items to assign.
        if int(graph.num_edges()) == 0:
            break

        # Rule 1: integer edge to virtual agent.
        chosen = find_virtual_agent_with_x_eq_one(graph, x_min)
        if chosen is not None:
            agent, item = chosen
            _assign_and_delete_item(agent, item)
            step += 1
            continue

        # Rule 2: agent only with leaf items.
        agent_only_with_leaf_items = find_agent_with_only_leaf_items(graph)
        if agent_only_with_leaf_items is not None:
            for _, item, _ in instance_copy.bids.bids_from_agent_id(agent_only_with_leaf_items):
                _assign_and_delete_item(agent_only_with_leaf_items, item)
            # instance_copy.bids.remove_agent(i_only_leaf)
            # instance_copy.budgets.remove(i_only_leaf)
            step += 1
            continue

        # Rule 3: tight leaf agent.
        tight_leaf_agent = find_tight_leaf_agent(graph)
        if tight_leaf_agent is None:
            raise RuntimeError(f"No rule can be applied in step {step}.")

        items_owned_by_other_agents = {
            item for (agent, item) in x.keys() if agent != tight_leaf_agent
        }
        for _, item, _ in instance_copy.bids.bids_from_agent_id(tight_leaf_agent):
            # item is leaf item.
            if item not in items_owned_by_other_agents:
                _assign_and_delete_item(tight_leaf_agent, item)

        # after deleting leaf items owned by tight leaf agent, there must be only one item left.
        items_left = instance_copy.bids.bids_from_agent_id(tight_leaf_agent)
        if len(items_left) != 1:
            raise RuntimeError(
                f"Agent {tight_leaf_agent} has {len(items_left)} Items left, expected 1."
            )

        # Set bid and budget of last item according to rule 3.
        _, item_left, _ = items_left[0]
        b_ij = instance.bids.get(tight_leaf_agent, item_left)
        x_ij = x.get((tight_leaf_agent, item_left), 0.0)
        new_b = (4.0 / 3.0) * b_ij * x_ij
        instance_copy.budgets.set(tight_leaf_agent, new_b)
        instance_copy.bids.set(tight_leaf_agent, item_left, new_b)
        virtual_agents[tight_leaf_agent] = True

        step += 1

    # Check if revenue is at least 3/4 of optimal lp revenue.
    if start_lp_opt > 0.0:
        revenue = allocation.total_revenue()
        need = apx_factor * start_lp_opt
        if revenue < need:
            raise RuntimeError(
                f"Value of allocation is too low."
                f"LP-Startvalue {start_lp_opt:.6f}, value has to be {need:.6f}, but is "
                f"{revenue:.6f}. "
            )

    return allocation, revenue, start_lp_opt
