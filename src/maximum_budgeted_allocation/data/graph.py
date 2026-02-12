from typing import Dict, Optional, Tuple

from graph_tool import Graph


def _opposite_vertex(e, v):
    if e.source() == v:
        return e.target()
    if e.target() == v:
        return e.source()
    raise ValueError("Vertex v is not incident to edge e")


def create_graph_from_instance(
    instance,
    x,
    x_min: float = 1e-12,
    virtual_agents: Dict[int, float] | None = None,
    tight_tol: float = 1e-12,
) -> Graph:
    if virtual_agents is None:
        virtual_agents = {}

    G = Graph(directed=False)

    v_bip = G.new_vertex_property("int")
    v_agent_idx = G.new_vertex_property("int")
    v_item_idx = G.new_vertex_property("int")
    v_is_tight = G.new_vertex_property("bool")
    v_is_virtual = G.new_vertex_property("bool")
    v_only_leaf = G.new_vertex_property("bool")
    v_is_leaf_agent = G.new_vertex_property("bool")

    e_x = G.new_edge_property("double")

    agent_v: Dict[int, int] = {}
    item_v: Dict[int, int] = {}

    for (i, j), xij in x.items():
        if xij <= x_min:
            continue
        b = instance.bids.get(i, j)
        if b <= 0.0:
            continue

        if i not in agent_v:
            v = G.add_vertex()
            v_bip[v] = 0
            v_agent_idx[v] = int(i)
            v_is_virtual[v] = bool(virtual_agents.get(int(i), False))
            v_is_tight[v] = False
            v_only_leaf[v] = False
            v_is_leaf_agent[v] = False
            agent_v[int(i)] = int(v)

        if j not in item_v:
            v = G.add_vertex()
            v_bip[v] = 1
            v_item_idx[v] = int(j)
            item_v[int(j)] = int(v)

    spent_per_agent: Dict[int, float] = {}
    for (i, j), xij in x.items():
        if xij <= x_min:
            continue
        b = instance.bids.get(i, j)
        if b <= 0.0:
            continue
        vi = agent_v[int(i)]
        vj = item_v[int(j)]
        e = G.add_edge(vi, vj)
        e_x[e] = float(xij)
        pay = float(xij) * float(b)
        spent_per_agent[int(i)] = spent_per_agent.get(int(i), 0.0) + pay

    if G.num_edges() == 0:
        G.vp["bipartite"] = v_bip
        G.vp["agent_index"] = v_agent_idx
        G.vp["item_index"] = v_item_idx
        G.vp["is_tight"] = v_is_tight
        G.vp["is_virtual"] = v_is_virtual
        G.vp["has_only_leaf_items"] = v_only_leaf
        G.vp["is_leaf_agent"] = v_is_leaf_agent
        G.ep["x"] = e_x
        return G

    leaf_items = {v for v in item_v.values() if G.vertex(v).out_degree() == 1}

    for i, v_int in agent_v.items():
        v = G.vertex(v_int)

        B_i = float(instance.budgets.get(i))
        used = float(spent_per_agent.get(int(i), 0.0))
        tight = used >= B_i - tight_tol
        v_is_tight[v] = tight

        total = 0
        leaf = 0
        for e in v.out_edges():
            u = _opposite_vertex(e, v)
            if int(v_bip[u]) != 1:
                continue
            total += 1
            if int(u) in leaf_items:
                leaf += 1

        nonleaf = total - leaf
        v_only_leaf[v] = total > 0 and nonleaf == 0

        is_leaf = total > 0 and leaf >= 1 and nonleaf <= 1
        v_is_leaf_agent[v] = is_leaf

    G.vp["bipartite"] = v_bip
    G.vp["agent_index"] = v_agent_idx
    G.vp["item_index"] = v_item_idx
    G.vp["is_tight"] = v_is_tight
    G.vp["is_virtual"] = v_is_virtual
    G.vp["has_only_leaf_items"] = v_only_leaf
    G.vp["is_leaf_agent"] = v_is_leaf_agent

    G.ep["x"] = e_x

    return G


def find_tight_leaf_agent(G: Graph) -> Optional[int]:
    v_bip = G.vp["bipartite"]
    v_is_tight = G.vp["is_tight"]
    v_is_leaf_agent = G.vp["is_leaf_agent"]
    v_agent_idx = G.vp["agent_index"]
    for v in G.vertices():
        if int(v_bip[v]) == 0 and bool(v_is_tight[v]) and bool(v_is_leaf_agent[v]):
            return int(v_agent_idx[v])
    return None


def find_agent_with_only_leaf_items(G: Graph) -> Optional[int]:
    v_bip = G.vp["bipartite"]
    v_only_leaf = G.vp["has_only_leaf_items"]
    v_agent_idx = G.vp["agent_index"]
    for v in G.vertices():
        if int(v_bip[v]) == 0 and bool(v_only_leaf[v]):
            return int(v_agent_idx[v])
    return None


def find_virtual_agent_with_x_eq_one(G: Graph, tol: float = 1e-12) -> Optional[Tuple[int, int]]:
    v_bip = G.vp["bipartite"]
    v_is_virtual = G.vp["is_virtual"]
    v_agent_idx = G.vp["agent_index"]
    v_item_idx = G.vp["item_index"]
    e_x = G.ep["x"]
    for v in G.vertices():
        if int(v_bip[v]) != 0 or not bool(v_is_virtual[v]):
            continue
        for e in v.out_edges():
            if abs(float(e_x[e]) - 1.0) <= tol:
                u = _opposite_vertex(e, v)
                if int(v_bip[u]) == 1:
                    return int(v_agent_idx[v]), int(v_item_idx[u])
    return None
