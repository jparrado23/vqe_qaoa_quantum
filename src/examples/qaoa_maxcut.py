"""
QAOA MaxCut example.

Demonstrates solving the MaxCut problem using QAOA with:
- Random graph generation
- Classical baseline (exact ILP solution)
- Quantum optimization
- Result comparison and visualization
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit.primitives import StatevectorSampler
from scipy.optimize import minimize

# Local imports
from scripts.hamiltonian import maxcut_hamiltonian
from scripts.qaoa_ansatz import create_qaoa_circuit, evaluate_bitstring, get_top_solutions
from utils.graph_utils import (
    create_random_graph, 
    solve_maxcut_classical, 
    evaluate_cut_value,
    bitstring_to_partition
)
from config.settings import (
    QAOA_CIRCUIT_CONFIG,
    QAOA_OPTIMIZER_CONFIG,
    QAOA_PROBLEM_CONFIG,
    PLOT_CONFIG
)


def qaoa_objective(params, circuit, hamiltonian, sampler, shots=1024):
    """
    Objective function for QAOA optimization.
    
    Uses sampling to estimate energy expectation value.
    """
    # Split parameters into gamma and beta
    p = len(params) // 2
    gamma = params[:p]
    beta = params[p:]
    
    # Bind parameters
    param_dict = {}
    for i in range(p):
        param_dict[circuit.parameters[2*i]] = gamma[i]
        param_dict[circuit.parameters[2*i + 1]] = beta[i]
    
    bound_circuit = circuit.assign_parameters(param_dict)
    
    # Sample
    job = sampler.run([bound_circuit], shots=shots)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    
    # Calculate energy from counts
    energy = 0.0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Calculate energy for this bitstring
        prob = count / total_shots
        
        # Evaluate Hamiltonian for this bitstring
        # Convert bitstring to statevector basis expectation
        bitstring_energy = 0.0
        for pauli_str, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            # Count number of 1s at Z positions (parity)
            parity = 0
            for idx, pauli_op in enumerate(str(pauli_str)[::-1]):
                if pauli_op == 'Z' and bitstring[idx] == '1':
                    parity += 1
            
            # Eigenvalue is (-1)^parity
            eigenvalue = (-1) ** parity
            bitstring_energy += np.real(coeff) * eigenvalue
        
        energy += prob * bitstring_energy
    
    return energy


def run_qaoa_maxcut(
    graph,
    p=1,
    shots=1024,
    max_iter=100,
    verbose=True
):
    """
    Run QAOA to solve MaxCut problem.
    
    Parameters
    ----------
    graph : Graph
        Input graph (networkx or rustworkx)
    p : int
        Number of QAOA layers
    shots : int
        Measurement shots per evaluation
    max_iter : int
        Maximum optimizer iterations
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Results including optimal parameters, energy, and top solutions
    """
    # Build cost Hamiltonian
    H_cost = maxcut_hamiltonian(graph)
    num_qubits = H_cost.num_qubits
    
    if verbose:
        print(f"Graph: {num_qubits} nodes")
        print(f"QAOA layers: p={p}")
        print(f"Parameters to optimize: {2*p}")
        print(f"Shots per evaluation: {shots}\n")
    
    # Create QAOA circuit
    circuit, gamma, beta = create_qaoa_circuit(H_cost, p=p)
    circuit.measure_all()
    
    # Initialize parameters randomly
    init_params = np.random.uniform(0, 2*np.pi, size=2*p)
    
    # Create sampler
    sampler = StatevectorSampler()
    
    # Track optimization progress
    iteration_count = [0]
    energy_history = []
    
    def objective(params):
        iteration_count[0] += 1
        energy = qaoa_objective(params, circuit, H_cost, sampler, shots=shots)
        energy_history.append(energy)
        
        if verbose and (iteration_count[0] % 10 == 0 or iteration_count[0] == 1):
            print(f"Iteration {iteration_count[0]:3d}: Energy = {energy:+.6f}")
        
        return energy
    
    # Run optimization
    if verbose:
        print("Starting QAOA optimization...\n")
    
    result = minimize(
        objective,
        init_params,
        method='COBYLA',
        options={'maxiter': max_iter}
    )
    
    optimal_params = result.x
    optimal_energy = result.fun
    
    if verbose:
        print(f"\nOptimization complete!")
        print(f"Optimal energy: {optimal_energy:.6f}")
        print(f"Converged: {result.success}")
        print(f"Iterations: {iteration_count[0]}\n")
    
    # Get final measurement results with optimal parameters
    p_opt = len(optimal_params) // 2
    param_dict = {}
    for i in range(p_opt):
        param_dict[circuit.parameters[2*i]] = optimal_params[i]
        param_dict[circuit.parameters[2*i + 1]] = optimal_params[p_opt + i]
    
    final_circuit = circuit.assign_parameters(param_dict)
    
    job = sampler.run([final_circuit], shots=10000)
    final_counts = job.result()[0].data.meas.get_counts()
    
    # Extract edge list for evaluation
    if hasattr(graph, 'edge_list'):
        edges = list(graph.edge_list())
    else:
        import networkx as nx
        mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        edges = [(mapping[e[0]], mapping[e[1]]) for e in graph.edges()]
    
    # Get top solutions
    top_solutions = get_top_solutions(final_counts, edges, top_k=20)
    
    return {
        'optimal_params': optimal_params,
        'optimal_energy': optimal_energy,
        'energy_history': energy_history,
        'final_counts': final_counts,
        'top_solutions': top_solutions,
        'num_iterations': iteration_count[0],
        'success': result.success
    }


def main():
    """Main execution function."""
    print("=" * 70)
    print("QAOA MaxCut Example")
    print("=" * 70)
    print()
    
    # Load configuration
    num_nodes = QAOA_PROBLEM_CONFIG['num_nodes']
    edge_prob = QAOA_PROBLEM_CONFIG['edge_probability']
    seed = QAOA_PROBLEM_CONFIG['seed']
    p = QAOA_CIRCUIT_CONFIG['p']
    
    # Create random graph
    print(f"Creating random graph...")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edge probability: {edge_prob}")
    print(f"  Seed: {seed}\n")
    
    graph = create_random_graph(num_nodes, edge_prob, seed=seed)
    
    num_edges = len(graph.edges())
    print(f"Generated graph with {num_edges} edges\n")
    
    # Solve classically
    print("Solving classically with Integer Linear Programming...")
    classical_value, classical_partition = solve_maxcut_classical(graph)
    print(f"Classical optimal MaxCut value: {classical_value}\n")
    
    # Run QAOA
    print("-" * 70)
    print("Running QAOA")
    print("-" * 70)
    print()
    
    qaoa_result = run_qaoa_maxcut(
        graph,
        p=p,
        shots=QAOA_OPTIMIZER_CONFIG.get('shots', 1024),
        max_iter=QAOA_OPTIMIZER_CONFIG['max_iter'],
        verbose=True
    )
    
    # Analyze results
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()
    
    # Get best QAOA solution
    best_qaoa_solution = qaoa_result['top_solutions'][0]
    best_bitstring, best_cut_value, best_count = best_qaoa_solution
    
    print(f"Classical Optimal: {classical_value} cuts")
    print(f"QAOA Best:         {best_cut_value} cuts")
    print(f"Approximation ratio: {best_cut_value / classical_value:.3f}")
    print()
    
    # Show top 5 solutions
    print("Top 5 QAOA solutions:")
    for i, (bitstring, cut_val, count) in enumerate(qaoa_result['top_solutions'][:5], 1):
        prob = count / sum(qaoa_result['final_counts'].values())
        print(f"  {i}. {bitstring}: {cut_val} cuts (prob: {prob:.3f})")
    print()
    
    # Plot convergence
    plt.figure(figsize=PLOT_CONFIG['figsize'])
    plt.plot(qaoa_result['energy_history'], 'b-', linewidth=2)
    plt.axhline(y=-classical_value/2, color='r', linestyle='--', 
                label=f'Classical optimal (E={-classical_value/2:.2f})')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title(f'QAOA Convergence (p={p}, {num_nodes} nodes)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if PLOT_CONFIG.get('save_plots', False):
        os.makedirs(PLOT_CONFIG['plot_dir'], exist_ok=True)
        plt.savefig(f"{PLOT_CONFIG['plot_dir']}qaoa_maxcut_convergence.png", 
                    dpi=PLOT_CONFIG['dpi'])
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
