"""
Hamiltonian construction module.

Functions for constructing quantum Hamiltonians for:
- Many-body spin systems (VQE)
- Combinatorial optimization problems (QAOA)
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
try:
    import networkx as nx
    import rustworkx as rx
    HAS_GRAPH_LIBS = True
except ImportError:
    HAS_GRAPH_LIBS = False


def heisenberg_xxz_2q(J: float = 1.0, delta: float = 1.0) -> SparsePauliOp:
    """
    Construct the XXZ Heisenberg Hamiltonian for 2 qubits.
    
    The Hamiltonian describes nearest-neighbor spin-1/2 interactions:
    
        H = J(σ_x ⊗ σ_x + σ_y ⊗ σ_y + Δ·σ_z ⊗ σ_z)
    
    where σ_x, σ_y, σ_z are Pauli matrices and ⊗ denotes tensor product.
    
    Physical interpretation:
    - J > 0: Antiferromagnetic coupling (favors anti-aligned spins → singlet state)
    - J < 0: Ferromagnetic coupling (favors aligned spins → triplet state)
    - Δ = 1: Isotropic (XXX) model with full rotational symmetry
    - Δ ≠ 1: Anisotropic model breaking symmetry along z-axis
    - Δ < 1: Favors alignment in xy-plane (easy-plane anisotropy)
    - Δ > 1: Favors alignment along z-axis (easy-axis anisotropy)

    Parameters
    ----------
    J : float, default=1.0
        Coupling strength. Positive for antiferromagnetic, negative for ferromagnetic.
    delta : float, default=1.0
        Anisotropy parameter. Controls the relative strength of ZZ interaction.
        
    Returns
    -------
    SparsePauliOp
        The Hamiltonian as a sum of weighted Pauli strings:
        - "XX" term with coefficient J
        - "YY" term with coefficient J
        - "ZZ" term with coefficient J*delta
        
    Examples
    --------
    >>> # Isotropic antiferromagnetic Heisenberg model
    >>> H_iso = heisenberg_xxz_2q(J=1.0, delta=1.0)
    
    >>> # Anisotropic model with stronger z-coupling
    >>> H_aniso = heisenberg_xxz_2q(J=1.0, delta=2.0)
    
    >>> # Ferromagnetic XXZ model
    >>> H_ferro = heisenberg_xxz_2q(J=-1.0, delta=1.0)
    
    Notes
    -----
    - Ground state for J=1, Δ=1 is the singlet |ψ⟩ = (|01⟩ - |10⟩)/√2 
      with energy E₀ = -3J
    - This Hamiltonian is small enough to solve exactly classically,
      making it ideal for VQE validation
    - For real hardware, each Pauli term requires separate measurement
      in the appropriate basis (X, Y, or Z)
    """
    # Define Pauli strings for the three interaction terms
    paulis = ["XX", "YY", "ZZ"]
    
    # Set coefficients: J for XX and YY, J*delta for ZZ
    coeffs = [J, J, J * delta]
    
    # Create SparsePauliOp (efficient representation for Pauli sums)
    return SparsePauliOp(paulis, coeffs=np.asarray(coeffs, dtype=complex))


def maxcut_hamiltonian(graph, weight: float = 1.0) -> SparsePauliOp:
    """
    Construct the MaxCut Hamiltonian for a given graph.
    
    The MaxCut problem seeks to partition graph vertices into two sets
    such that the number (or total weight) of edges between the sets is maximized.
    
    Quantum formulation:
        H_MaxCut = -∑_{(i,j)∈E} w_{ij} * (I - Z_i Z_j) / 2
    
    Equivalently (dropping constant term I):
        H_MaxCut = (1/2) ∑_{(i,j)∈E} w_{ij} * Z_i Z_j
    
    We minimize this Hamiltonian. Each edge contributes:
    - Z_i Z_j = +1 if qubits i, j in same partition → not in cut
    - Z_i Z_j = -1 if qubits i, j in different partitions → in cut
    
    Thus minimizing H maximizes the cut value.
    
    Parameters
    ----------
    graph : networkx.Graph or rustworkx.PyGraph
        The graph for the MaxCut problem. Nodes are mapped to qubits.
        If graph has 'weight' edge attributes, they are used.
    weight : float, default=1.0
        Default edge weight if graph edges don't have 'weight' attribute.
        
    Returns
    -------
    SparsePauliOp
        The MaxCut Hamiltonian as a sum of ZZ Pauli terms.
        Number of qubits matches number of graph nodes.
        
    Examples
    --------
    >>> import networkx as nx
    >>> # Triangle graph
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0,1), (1,2), (2,0)])
    >>> H = maxcut_hamiltonian(G)
    >>> print(f"Hamiltonian has {H.num_qubits} qubits")
    
    >>> # Weighted graph
    >>> G_weighted = nx.Graph()
    >>> G_weighted.add_edge(0, 1, weight=2.0)
    >>> G_weighted.add_edge(1, 2, weight=1.5)
    >>> H_weighted = maxcut_hamiltonian(G_weighted)
    
    Notes
    -----
    - Graph nodes are automatically mapped to qubits 0, 1, ..., n-1
    - For rustworkx graphs, uses get_edge_data() for weights
    - For networkx graphs, uses edge attributes
    - Classical optimal solution can be found via ILP for validation
    
    References
    ----------
    Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
    arXiv:1411.4028
    """
    if not HAS_GRAPH_LIBS:
        raise ImportError(
            "Graph libraries required. Install with: "
            "pip install networkx rustworkx"
        )
    
    # Determine graph type and extract edge information
    if hasattr(graph, 'edge_list'):  # rustworkx.PyGraph
        num_qubits = len(graph.nodes())
        edges = graph.edge_list()
        
        # Build Pauli list for MaxCut Hamiltonian
        pauli_list = []
        for i, edge in enumerate(edges):
            # Create ZZ term for this edge
            paulis = ["I"] * num_qubits
            paulis[edge[0]] = "Z"
            paulis[edge[1]] = "Z"
            
            # Get edge weight
            edge_weight = graph.get_edge_data(edge[0], edge[1])
            if edge_weight is None:
                edge_weight = weight
            
            # Pauli string is reversed (Qiskit convention)
            pauli_list.append(("".join(paulis)[::-1], 0.5 * edge_weight))
            
    elif hasattr(graph, 'edges'):  # networkx.Graph
        num_qubits = len(graph.nodes())
        
        # Relabel nodes to 0, 1, ..., n-1 if needed
        mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        graph_relabeled = nx.relabel_nodes(graph, mapping)
        
        pauli_list = []
        for edge in graph_relabeled.edges():
            # Create ZZ term for this edge
            paulis = ["I"] * num_qubits
            paulis[edge[0]] = "Z"
            paulis[edge[1]] = "Z"
            
            # Get edge weight
            edge_data = graph_relabeled.get_edge_data(*edge)
            edge_weight = edge_data.get('weight', weight) if edge_data else weight
            
            # Pauli string is reversed (Qiskit convention)
            pauli_list.append(("".join(paulis)[::-1], 0.5 * edge_weight))
    else:
        raise TypeError(
            "graph must be networkx.Graph or rustworkx.PyGraph"
        )
    
    return SparsePauliOp.from_list(pauli_list)


def build_max_cut_paulis(graph) -> list:
    """
    Convert graph to Pauli list representation for MaxCut.
    
    Helper function that returns the raw Pauli list before creating
    SparsePauliOp. Useful for custom Hamiltonian manipulation.
    
    Parameters
    ----------
    graph : networkx.Graph or rustworkx.PyGraph
        The graph for the MaxCut problem
        
    Returns
    -------
    list[tuple[str, float]]
        List of (pauli_string, coefficient) tuples
        
    Examples
    --------
    >>> G = nx.complete_graph(4)
    >>> paulis = build_max_cut_paulis(G)
    >>> # Manually create Hamiltonian with custom coefficients
    >>> H = SparsePauliOp.from_list(paulis)
    """
    if not HAS_GRAPH_LIBS:
        raise ImportError(
            "Graph libraries required. Install with: "
            "pip install networkx rustworkx"
        )
    
    if hasattr(graph, 'edge_list'):  # rustworkx
        pauli_list = []
        num_nodes = len(graph.nodes())
        
        for edge in graph.edge_list():
            paulis = ["I"] * num_nodes
            paulis[edge[0]] = "Z"
            paulis[edge[1]] = "Z"
            
            weight = graph.get_edge_data(edge[0], edge[1])
            if weight is None:
                weight = 1.0
                
            pauli_list.append(("".join(paulis)[::-1], 0.5 * weight))
            
    else:  # networkx
        pauli_list = []
        num_nodes = len(graph.nodes())
        mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        
        for edge in graph.edges():
            u, v = mapping[edge[0]], mapping[edge[1]]
            paulis = ["I"] * num_nodes
            paulis[u] = "Z"
            paulis[v] = "Z"
            
            edge_data = graph.get_edge_data(*edge)
            weight = edge_data.get('weight', 1.0) if edge_data else 1.0
            
            pauli_list.append(("".join(paulis)[::-1], 0.5 * weight))
    
    return pauli_list

