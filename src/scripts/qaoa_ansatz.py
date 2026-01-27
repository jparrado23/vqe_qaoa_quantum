"""
QAOA ansatz construction module.

Functions for constructing Quantum Approximate Optimization Algorithm (QAOA)
circuits for combinatorial optimization problems.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


def create_qaoa_circuit(
    cost_hamiltonian: SparsePauliOp,
    p: int = 1,
    mixer_hamiltonian: SparsePauliOp = None,
    initial_state: QuantumCircuit = None
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Create a QAOA circuit for a given cost Hamiltonian.
    
    QAOA alternates between:
    1. Cost layer: e^(-iγ_j H_C) encodes the problem
    2. Mixer layer: e^(-iβ_j H_M) enables exploration
    
    This is repeated p times with different parameters (γ, β) at each layer.
    
    Parameters
    ----------
    cost_hamiltonian : SparsePauliOp
        The problem Hamiltonian to minimize (e.g., MaxCut)
    p : int, default=1
        Number of QAOA layers. Higher p allows better approximation
        but increases circuit depth and parameter count.
    mixer_hamiltonian : SparsePauliOp, optional
        Mixer Hamiltonian. If None, uses standard X-mixer:
        H_M = ∑_i X_i (transverse field)
    initial_state : QuantumCircuit, optional
        Initial state preparation. If None, uses |+⟩^⊗n
        (uniform superposition via Hadamard gates)
        
    Returns
    -------
    circuit : QuantumCircuit
        The QAOA circuit with 2p parameters
    gamma : ParameterVector
        Cost layer parameters (length p)
    beta : ParameterVector
        Mixer layer parameters (length p)
        
    Examples
    --------
    >>> import networkx as nx
    >>> from scripts.hamiltonian import maxcut_hamiltonian
    >>> 
    >>> # Create MaxCut Hamiltonian for triangle graph
    >>> G = nx.cycle_graph(3)
    >>> H_cost = maxcut_hamiltonian(G)
    >>> 
    >>> # Build QAOA circuit with p=2 layers
    >>> qaoa_circ, gamma, beta = create_qaoa_circuit(H_cost, p=2)
    >>> print(f"Circuit has {qaoa_circ.num_parameters} parameters")
    Circuit has 4 parameters
    
    >>> # Bind specific parameter values
    >>> params = {gamma[0]: 0.5, beta[0]: 1.2, gamma[1]: 0.3, beta[1]: 0.8}
    >>> bound_circ = qaoa_circ.assign_parameters(params)
    
    Notes
    -----
    - Standard QAOA uses X-mixer: all qubits are mixed independently
    - For constrained problems, custom mixers preserve feasibility
    - p=1 often gives good approximation ratios (>0.6 for MaxCut)
    - Optimal (γ*, β*) typically found via classical optimization
    
    References
    ----------
    Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
    arXiv:1411.4028
    """
    num_qubits = cost_hamiltonian.num_qubits
    
    # Create empty mixer if not provided (will use default X-mixer)
    if mixer_hamiltonian is None:
        mixer = QuantumCircuit(num_qubits)
    else:
        mixer = mixer_hamiltonian
    
    # Create empty initial state if not provided (will use |+⟩^⊗n)
    if initial_state is None:
        init_state = QuantumCircuit(num_qubits)
    else:
        init_state = initial_state
    
    # Use Qiskit's built-in QAOAAnsatz
    ansatz = QAOAAnsatz(
        cost_operator=cost_hamiltonian,
        reps=p,
        initial_state=init_state if initial_state else None,
        mixer_operator=mixer if mixer_hamiltonian else None
    )
    
    # Extract parameter vectors for easy access
    # QAOAAnsatz creates parameters in order: [γ_0, β_0, γ_1, β_1, ...]
    params = list(ansatz.parameters)
    gamma = ParameterVector("γ", p)
    beta = ParameterVector("β", p)
    
    # Create parameter mapping
    param_dict = {}
    for i in range(p):
        param_dict[params[2*i]] = gamma[i]      # Cost parameters
        param_dict[params[2*i + 1]] = beta[i]   # Mixer parameters
    
    # Rebind with clearer parameter names
    ansatz = ansatz.assign_parameters(param_dict)
    
    return ansatz, gamma, beta


def create_simple_qaoa_circuit(
    num_qubits: int,
    edges: list,
    p: int = 1
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Create a simple QAOA circuit for MaxCut directly from edges.
    
    Simplified interface that builds the circuit without requiring
    explicit Hamiltonian construction. Useful for quick prototyping.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits (vertices in graph)
    edges : list[tuple[int, int]]
        List of edges as (node1, node2) tuples
    p : int, default=1
        Number of QAOA layers
        
    Returns
    -------
    circuit : QuantumCircuit
        The QAOA circuit
    gamma : ParameterVector
        Cost parameters
    beta : ParameterVector
        Mixer parameters
        
    Examples
    --------
    >>> # Triangle graph: 3 vertices, 3 edges
    >>> edges = [(0, 1), (1, 2), (2, 0)]
    >>> circuit, gamma, beta = create_simple_qaoa_circuit(3, edges, p=1)
    """
    qc = QuantumCircuit(num_qubits)
    
    # Create parameter vectors
    gamma = ParameterVector("γ", p)
    beta = ParameterVector("β", p)
    
    # Initial state: |+⟩^⊗n
    qc.h(range(num_qubits))
    
    # QAOA layers
    for layer in range(p):
        # Cost layer: ZZ rotations for each edge
        for edge in edges:
            i, j = edge[0], edge[1]
            qc.cx(i, j)
            qc.rz(2 * gamma[layer], j)
            qc.cx(i, j)
        
        # Mixer layer: X rotations on all qubits
        for qubit in range(num_qubits):
            qc.rx(2 * beta[layer], qubit)
    
    return qc, gamma, beta


def qaoa_circuit_with_measurements(
    cost_hamiltonian: SparsePauliOp,
    p: int = 1
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Create QAOA circuit with measurements for sampling.
    
    Adds measurement operations to the circuit, preparing it for
    execution on quantum hardware or shot-based simulators.
    
    Parameters
    ----------
    cost_hamiltonian : SparsePauliOp
        The cost Hamiltonian
    p : int, default=1
        Number of QAOA layers
        
    Returns
    -------
    circuit : QuantumCircuit
        QAOA circuit with measurements
    gamma : ParameterVector
        Cost parameters
    beta : ParameterVector
        Mixer parameters
        
    Examples
    --------
    >>> H = maxcut_hamiltonian(graph)
    >>> circuit, gamma, beta = qaoa_circuit_with_measurements(H, p=2)
    >>> # Circuit is ready for Sampler primitive
    """
    circuit, gamma, beta = create_qaoa_circuit(cost_hamiltonian, p)
    circuit.measure_all()
    return circuit, gamma, beta


def evaluate_bitstring(bitstring: str, edges: list) -> int:
    """
    Evaluate MaxCut objective value for a given bitstring.
    
    Counts how many edges are "cut" (have endpoints in different partitions).
    
    Parameters
    ----------
    bitstring : str
        Binary string representing partition (e.g., "0110")
        '0' = partition A, '1' = partition B
    edges : list[tuple[int, int]]
        List of edges
        
    Returns
    -------
    int
        Number of cut edges
        
    Examples
    --------
    >>> edges = [(0, 1), (1, 2), (2, 0)]  # Triangle
    >>> evaluate_bitstring("010", edges)  # Vertex 1 separated
    2
    >>> evaluate_bitstring("000", edges)  # All same partition
    0
    """
    cut_value = 0
    for edge in edges:
        i, j = edge[0], edge[1]
        # Edge is cut if endpoints have different values
        if bitstring[i] != bitstring[j]:
            cut_value += 1
    return cut_value


def get_top_solutions(counts: dict, edges: list, top_k: int = 10) -> list:
    """
    Extract top-k solutions from measurement counts.
    
    Ranks bitstrings by their MaxCut objective value.
    
    Parameters
    ----------
    counts : dict
        Measurement counts from Sampler
    edges : list[tuple[int, int]]
        Graph edges
    top_k : int, default=10
        Number of top solutions to return
        
    Returns
    -------
    list[tuple[str, int, int]]
        List of (bitstring, cut_value, count) sorted by cut_value
        
    Examples
    --------
    >>> counts = {'010': 523, '101': 489, '000': 12}
    >>> edges = [(0,1), (1,2), (2,0)]
    >>> top_solutions = get_top_solutions(counts, edges, top_k=2)
    >>> for bitstring, cut_val, cnt in top_solutions:
    ...     print(f"{bitstring}: {cut_val} cuts ({cnt} times)")
    """
    # Evaluate all bitstrings
    solutions = []
    for bitstring, count in counts.items():
        cut_value = evaluate_bitstring(bitstring, edges)
        solutions.append((bitstring, cut_value, count))
    
    # Sort by cut value (descending)
    solutions.sort(key=lambda x: x[1], reverse=True)
    
    return solutions[:top_k]
