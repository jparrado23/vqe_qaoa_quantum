"""
Graph utilities for QAOA optimization problems.

Functions for creating, manipulating, and solving classical graph problems
to compare with quantum results.
"""

import numpy as np
try:
    import networkx as nx
    import rustworkx as rx
    HAS_NETWORKX = True
    HAS_RUSTWORKX = True
except ImportError:
    HAS_NETWORKX = False
    HAS_RUSTWORKX = False

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def create_random_graph(
    num_nodes: int,
    edge_probability: float = 0.5,
    seed: int = None,
    use_rustworkx: bool = False
):
    """
    Create a random Erdős-Rényi graph.
    
    Parameters
    ----------
    num_nodes : int
        Number of vertices
    edge_probability : float, default=0.5
        Probability of edge between any two nodes
    seed : int, optional
        Random seed for reproducibility
    use_rustworkx : bool, default=False
        If True, return rustworkx.PyGraph, else networkx.Graph
        
    Returns
    -------
    Graph
        Random graph (networkx or rustworkx)
        
    Examples
    --------
    >>> G = create_random_graph(10, edge_probability=0.3, seed=42)
    >>> print(f"Graph has {len(G.edges())} edges")
    """
    if seed is not None:
        np.random.seed(seed)
    
    if use_rustworkx:
        if not HAS_RUSTWORKX:
            raise ImportError("rustworkx required. Install with: pip install rustworkx")
        
        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_nodes))
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < edge_probability:
                    graph.add_edge(i, j, 1.0)
        return graph
    else:
        if not HAS_NETWORKX:
            raise ImportError("networkx required. Install with: pip install networkx")
        
        return nx.erdos_renyi_graph(num_nodes, edge_probability, seed=seed)


def create_regular_graph(
    num_nodes: int,
    degree: int,
    seed: int = None,
    use_rustworkx: bool = False
):
    """
    Create a random regular graph (all nodes have same degree).
    
    Parameters
    ----------
    num_nodes : int
        Number of vertices
    degree : int
        Degree of each vertex. Must satisfy num_nodes * degree is even.
    seed : int, optional
        Random seed
    use_rustworkx : bool, default=False
        Return type selection
        
    Returns
    -------
    Graph
        Random regular graph
        
    Examples
    --------
    >>> G = create_regular_graph(10, degree=3, seed=42)
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required for regular graphs")
    
    graph = nx.random_regular_graph(degree, num_nodes, seed=seed)
    
    if use_rustworkx:
        # Convert to rustworkx
        rx_graph = rx.PyGraph()
        rx_graph.add_nodes_from(range(num_nodes))
        for edge in graph.edges():
            rx_graph.add_edge(edge[0], edge[1], 1.0)
        return rx_graph
    
    return graph


def create_grid_graph(rows: int, cols: int, use_rustworkx: bool = False):
    """
    Create a 2D grid graph.
    
    Parameters
    ----------
    rows : int
        Number of rows
    cols : int
        Number of columns
    use_rustworkx : bool, default=False
        Return type selection
        
    Returns
    -------
    Graph
        Grid graph with rows * cols vertices
        
    Examples
    --------
    >>> G = create_grid_graph(3, 4)  # 3x4 grid, 12 vertices
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required")
    
    graph = nx.grid_2d_graph(rows, cols)
    # Relabel nodes to integers
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    
    if use_rustworkx:
        rx_graph = rx.PyGraph()
        rx_graph.add_nodes_from(range(len(graph.nodes())))
        for edge in graph.edges():
            rx_graph.add_edge(edge[0], edge[1], 1.0)
        return rx_graph
    
    return graph


def graph_from_coupling_map(coupling_map, num_qubits: int):
    """
    Create graph from hardware coupling map.
    
    Useful for creating MaxCut problems that match hardware topology.
    
    Parameters
    ----------
    coupling_map : CouplingMap or list
        Hardware coupling map
    num_qubits : int
        Number of qubits to use
        
    Returns
    -------
    rustworkx.PyGraph
        Graph matching hardware connectivity
        
    Examples
    --------
    >>> from qiskit.providers.fake_provider import GenericBackendV2
    >>> backend = GenericBackendV2(num_qubits=27)
    >>> G = graph_from_coupling_map(backend.coupling_map, num_qubits=10)
    """
    if not HAS_RUSTWORKX:
        raise ImportError("rustworkx required")
    
    graph = rx.PyGraph()
    graph.add_nodes_from(range(num_qubits))
    
    edges_added = set()
    for edge in coupling_map:
        if edge[0] < num_qubits and edge[1] < num_qubits:
            # Avoid duplicate edges
            edge_tuple = tuple(sorted([edge[0], edge[1]]))
            if edge_tuple not in edges_added:
                graph.add_edge(edge[0], edge[1], 1.0)
                edges_added.add(edge_tuple)
    
    return graph


def solve_maxcut_classical(graph) -> tuple[int, list]:
    """
    Solve MaxCut exactly using Integer Linear Programming.
    
    Provides the classical optimal solution for comparison with QAOA.
    
    Mathematical formulation:
        maximize: ∑_{(i,j)∈E} w_ij * y_ij
        subject to:
            y_ij <= x_i + x_j              ∀(i,j)∈E
            y_ij <= 2 - x_i - x_j          ∀(i,j)∈E
            x_i ∈ {0, 1}                   ∀i∈V
            y_ij ∈ {0, 1}                  ∀(i,j)∈E
    
    Where x_i indicates partition membership and y_ij indicates if edge is cut.
    
    Parameters
    ----------
    graph : networkx.Graph or rustworkx.PyGraph
        The input graph
        
    Returns
    -------
    optimal_value : int
        Maximum cut value
    optimal_partition : list[int]
        Optimal partition as list of 0s and 1s
        
    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.cycle_graph(4)
    >>> optimal_val, partition = solve_maxcut_classical(G)
    >>> print(f"Optimal MaxCut value: {optimal_val}")
    Optimal MaxCut value: 4
    
    Notes
    -----
    - Uses PuLP for ILP solving (requires: pip install pulp)
    - Exponential worst-case complexity (NP-hard problem)
    - Practical for graphs up to ~100 nodes
    - For larger graphs, use approximation algorithms
    """
    if not HAS_PULP:
        raise ImportError(
            "PuLP required for classical MaxCut solving. "
            "Install with: pip install pulp"
        )
    
    # Handle rustworkx graphs
    if hasattr(graph, 'edge_list'):
        edges = list(graph.edge_list())
        nodes = list(range(len(graph.nodes())))
        
        def get_weight(edge):
            w = graph.get_edge_data(edge[0], edge[1])
            return w if w is not None else 1.0
    
    # Handle networkx graphs
    else:
        # Relabel to 0, 1, ..., n-1
        mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        graph_relabeled = nx.relabel_nodes(graph, mapping)
        
        edges = list(graph_relabeled.edges())
        nodes = list(graph_relabeled.nodes())
        
        def get_weight(edge):
            data = graph_relabeled.get_edge_data(*edge)
            return data.get('weight', 1.0) if data else 1.0
    
    # Create ILP problem
    prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)
    
    # Decision variables: x[i] = 1 if node i in partition B
    x = pulp.LpVariable.dicts("x", nodes, cat="Binary")
    
    # Edge variables: y[(i,j)] = 1 if edge (i,j) is cut
    y = pulp.LpVariable.dicts("y", edges, cat="Binary")
    
    # Objective: maximize total weight of cut edges
    prob += pulp.lpSum([get_weight(edge) * y[edge] for edge in edges])
    
    # Constraints: y[i,j] = 1 iff x[i] != x[j]
    for edge in edges:
        i, j = edge[0], edge[1]
        # y_ij <= x_i + x_j (if both in A, can't be cut)
        prob += y[edge] <= x[i] + x[j]
        # y_ij <= 2 - x_i - x_j (if both in B, can't be cut)
        prob += y[edge] <= 2 - x[i] - x[j]
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # Suppress solver output
    
    # Extract solution
    optimal_value = pulp.value(prob.objective)
    optimal_partition = [int(x[node].varValue) for node in nodes]
    
    return int(optimal_value), optimal_partition


def evaluate_cut_value(graph, partition: list) -> int:
    """
    Evaluate the cut value for a given partition.
    
    Parameters
    ----------
    graph : Graph
        The input graph
    partition : list[int]
        Partition as list of 0s and 1s
        
    Returns
    -------
    int
        Number of cut edges
        
    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> partition = [0, 1, 0, 1]  # Alternating partition
    >>> cut_val = evaluate_cut_value(G, partition)
    >>> print(cut_val)
    4
    """
    cut_value = 0
    
    if hasattr(graph, 'edge_list'):  # rustworkx
        for edge in graph.edge_list():
            if partition[edge[0]] != partition[edge[1]]:
                weight = graph.get_edge_data(edge[0], edge[1])
                cut_value += weight if weight is not None else 1.0
    else:  # networkx
        mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        for edge in graph.edges():
            u, v = mapping[edge[0]], mapping[edge[1]]
            if partition[u] != partition[v]:
                data = graph.get_edge_data(*edge)
                weight = data.get('weight', 1.0) if data else 1.0
                cut_value += weight
    
    return int(cut_value)


def bitstring_to_partition(bitstring: str) -> list:
    """
    Convert bitstring to partition list.
    
    Parameters
    ----------
    bitstring : str
        Binary string (e.g., "0110")
        
    Returns
    -------
    list[int]
        Partition as list of integers
        
    Examples
    --------
    >>> partition = bitstring_to_partition("0110")
    >>> print(partition)
    [0, 1, 1, 0]
    """
    return [int(bit) for bit in bitstring]


def partition_to_bitstring(partition: list) -> str:
    """
    Convert partition list to bitstring.
    
    Parameters
    ----------
    partition : list[int]
        Partition as list of 0s and 1s
        
    Returns
    -------
    str
        Binary string
        
    Examples
    --------
    >>> bitstring = partition_to_bitstring([0, 1, 1, 0])
    >>> print(bitstring)
    0110
    """
    return ''.join(str(bit) for bit in partition)
