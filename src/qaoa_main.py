#!/usr/bin/env python3
"""
QAOA Main Execution Script

Command-line interface for running QAOA (Quantum Approximate Optimization Algorithm)
for combinatorial optimization problems, primarily MaxCut.

Examples:
    # Run QAOA on random 12-node graph
    python qaoa_main.py
    
    # Custom graph size and parameters
    python qaoa_main.py --nodes 20 --p 3 --shots 2000
    
    # Different graph types
    python qaoa_main.py --graph-type regular --degree 3 --nodes 15
    python qaoa_main.py --graph-type grid --rows 4 --cols 4
    
    # Skip classical solver (for large graphs)
    python qaoa_main.py --nodes 50 --skip-classical
    
    # Hardware execution (requires IBM Quantum token)
    python qaoa_main.py --backend hardware --nodes 10 --p 2
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Qiskit imports
from qiskit.primitives import StatevectorSampler
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler
    HAS_IBM_RUNTIME = True
except ImportError:
    HAS_IBM_RUNTIME = False

# Local imports
from src.scripts.hamiltonian import maxcut_hamiltonian
from src.scripts.qaoa_ansatz import create_qaoa_circuit, get_top_solutions, evaluate_bitstring
from src.utils.graph_utils import (
    create_random_graph,
    create_regular_graph,
    create_grid_graph,
    solve_maxcut_classical,
    evaluate_cut_value,
)
from src.config.settings import (
    QAOA_CIRCUIT_CONFIG,
    QAOA_OPTIMIZER_CONFIG,
    QAOA_PROBLEM_CONFIG,
    QAOA_HARDWARE_CONFIG,
    PLOT_CONFIG,
)


def qaoa_objective(params, circuit, hamiltonian, sampler, shots=1024, verbose=False):
    """
    Objective function for QAOA optimization.
    
    Parameters
    ----------
    params : np.ndarray
        Current parameter values [γ_0, β_0, γ_1, β_1, ...]
    circuit : QuantumCircuit
        QAOA circuit with parameters
    hamiltonian : SparsePauliOp
        Cost Hamiltonian
    sampler : Sampler
        Qiskit sampler primitive
    shots : int
        Number of measurement shots
    verbose : bool
        Print detailed evaluation info
        
    Returns
    -------
    float
        Energy expectation value
    """
    # Split parameters into gamma and beta
    p = len(params) // 2
    gamma = params[:p]
    beta = params[p:]
    
    # Bind parameters to circuit
    param_dict = {}
    circuit_params = list(circuit.parameters)
    for i in range(p):
        param_dict[circuit_params[2*i]] = gamma[i]
        param_dict[circuit_params[2*i + 1]] = beta[i]
    
    bound_circuit = circuit.assign_parameters(param_dict)
    
    # Sample from circuit
    job = sampler.run([bound_circuit], shots=shots)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    
    # Calculate energy expectation value
    energy = 0.0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        prob = count / total_shots
        
        # Evaluate energy for this bitstring
        bitstring_energy = 0.0
        for pauli_str, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            # Count parity at Z positions
            parity = 0
            for idx, pauli_op in enumerate(str(pauli_str)[::-1]):
                if pauli_op == 'Z' and bitstring[idx] == '1':
                    parity += 1
            
            # Eigenvalue is (-1)^parity
            eigenvalue = (-1) ** parity
            bitstring_energy += np.real(coeff) * eigenvalue
        
        energy += prob * bitstring_energy
    
    if verbose:
        print(f"  Energy: {energy:.6f}")
    
    return energy


def run_qaoa(
    graph,
    p=1,
    shots=1024,
    max_iter=100,
    method='COBYLA',
    initial_params=None,
    sampler=None,
    verbose=True
):
    """
    Run QAOA optimization.
    
    Parameters
    ----------
    graph : Graph
        Problem graph
    p : int
        Number of QAOA layers
    shots : int
        Measurement shots per evaluation
    max_iter : int
        Maximum optimizer iterations
    method : str
        Optimization method
    initial_params : np.ndarray, optional
        Starting parameters
    sampler : Sampler, optional
        Qiskit sampler (creates StatevectorSampler if None)
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Optimization results
    """
    # Build cost Hamiltonian
    H_cost = maxcut_hamiltonian(graph)
    num_qubits = H_cost.num_qubits
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"QAOA Optimization")
        print(f"{'='*70}")
        print(f"Graph: {num_qubits} nodes")
        print(f"QAOA layers (p): {p}")
        print(f"Parameters to optimize: {2*p}")
        print(f"Shots per evaluation: {shots}")
        print(f"Optimizer: {method}")
        print(f"{'='*70}\n")
    
    # Create QAOA circuit
    circuit, gamma, beta = create_qaoa_circuit(H_cost, p=p)
    circuit.measure_all()
    
    # Initialize parameters
    if initial_params is None:
        initial_params = np.random.uniform(0, 2*np.pi, size=2*p)
    
    # Create sampler if not provided
    if sampler is None:
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
        print("Starting optimization...\n")
    
    result = minimize(
        objective,
        initial_params,
        method=method,
        options={'maxiter': max_iter}
    )
    
    optimal_params = result.x
    optimal_energy = result.fun
    
    if verbose:
        print(f"\n{'='*70}")
        print("Optimization Complete")
        print(f"{'='*70}")
        print(f"Optimal energy: {optimal_energy:.6f}")
        print(f"Converged: {result.success}")
        print(f"Total iterations: {iteration_count[0]}")
        print(f"{'='*70}\n")
    
    # Get final measurement results with optimal parameters
    p_opt = len(optimal_params) // 2
    param_dict = {}
    circuit_params = list(circuit.parameters)
    for i in range(p_opt):
        param_dict[circuit_params[2*i]] = optimal_params[i]
        param_dict[circuit_params[2*i + 1]] = optimal_params[p_opt + i]
    
    final_circuit = circuit.assign_parameters(param_dict)
    
    # Get high-statistics final measurement
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
        'success': result.success,
        'result': result,
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='QAOA for MaxCut Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (12-node random graph, p=5)
  python qaoa_main.py
  
  # Custom graph size and QAOA layers
  python qaoa_main.py --nodes 20 --p 3
  
  # Regular graph (all nodes same degree)
  python qaoa_main.py --graph-type regular --degree 3 --nodes 15 --p 4
  
  # Grid graph
  python qaoa_main.py --graph-type grid --rows 4 --cols 4 --p 2
  
  # Skip classical solver for large graphs
  python qaoa_main.py --nodes 50 --p 3 --skip-classical
  
  # More optimizer iterations
  python qaoa_main.py --max-iter 300 --shots 2048
        """
    )
    
    # Graph parameters
    graph_group = parser.add_argument_group('Graph Configuration')
    graph_group.add_argument(
        '--graph-type',
        type=str,
        default=QAOA_PROBLEM_CONFIG['graph_type'],
        choices=['random', 'regular', 'grid'],
        help='Type of graph to generate (default: random)'
    )
    graph_group.add_argument(
        '--nodes',
        type=int,
        default=QAOA_PROBLEM_CONFIG['num_nodes'],
        help='Number of nodes for random/regular graphs (default: 12)'
    )
    graph_group.add_argument(
        '--edge-prob',
        type=float,
        default=QAOA_PROBLEM_CONFIG['edge_probability'],
        help='Edge probability for random graphs (default: 0.5)'
    )
    graph_group.add_argument(
        '--degree',
        type=int,
        default=QAOA_PROBLEM_CONFIG['degree'],
        help='Degree for regular graphs (default: 3)'
    )
    graph_group.add_argument(
        '--rows',
        type=int,
        default=4,
        help='Number of rows for grid graphs (default: 4)'
    )
    graph_group.add_argument(
        '--cols',
        type=int,
        default=4,
        help='Number of columns for grid graphs (default: 4)'
    )
    graph_group.add_argument(
        '--seed',
        type=int,
        default=QAOA_PROBLEM_CONFIG['seed'],
        help='Random seed for reproducibility (default: 42)'
    )
    
    # QAOA parameters
    qaoa_group = parser.add_argument_group('QAOA Configuration')
    qaoa_group.add_argument(
        '--p',
        type=int,
        default=QAOA_CIRCUIT_CONFIG['p'],
        help='Number of QAOA layers (default: 5)'
    )
    qaoa_group.add_argument(
        '--shots',
        type=int,
        default=1024,
        help='Measurement shots per evaluation (default: 1024)'
    )
    
    # Optimization parameters
    opt_group = parser.add_argument_group('Optimization Configuration')
    opt_group.add_argument(
        '--method',
        type=str,
        default=QAOA_OPTIMIZER_CONFIG['method'],
        choices=['COBYLA', 'SLSQP', 'Powell', 'Nelder-Mead'],
        help='Classical optimizer (default: COBYLA)'
    )
    opt_group.add_argument(
        '--max-iter',
        type=int,
        default=QAOA_OPTIMIZER_CONFIG['max_iter'],
        help='Maximum optimizer iterations (default: 200)'
    )
    
    # Execution options
    exec_group = parser.add_argument_group('Execution Options')
    exec_group.add_argument(
        '--backend',
        type=str,
        default='statevector',
        choices=['statevector', 'hardware'],
        help='Execution backend (default: statevector)'
    )
    exec_group.add_argument(
        '--skip-classical',
        action='store_true',
        help='Skip classical ILP solver (for large graphs)'
    )
    exec_group.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting convergence'
    )
    exec_group.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Save results to file (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("QAOA for MaxCut Problem")
    print("="*70)
    
    # Generate graph
    print(f"\nGenerating {args.graph_type} graph...")
    if args.graph_type == 'random':
        graph = create_random_graph(
            args.nodes,
            edge_probability=args.edge_prob,
            seed=args.seed
        )
        print(f"  Nodes: {args.nodes}")
        print(f"  Edge probability: {args.edge_prob}")
    elif args.graph_type == 'regular':
        graph = create_regular_graph(
            args.nodes,
            degree=args.degree,
            seed=args.seed
        )
        print(f"  Nodes: {args.nodes}")
        print(f"  Degree: {args.degree}")
    else:  # grid
        graph = create_grid_graph(args.rows, args.cols)
        print(f"  Grid: {args.rows} x {args.cols}")
    
    num_edges = len(graph.edges())
    print(f"  Edges: {num_edges}")
    print(f"  Seed: {args.seed}")
    
    # Solve classically if requested
    classical_value = None
    classical_partition = None
    if not args.skip_classical:
        try:
            print("\nSolving classically with ILP...")
            classical_value, classical_partition = solve_maxcut_classical(graph)
            print(f"  Classical optimal: {classical_value} cuts")
        except Exception as e:
            print(f"  Classical solver failed: {e}")
            print("  Continuing without classical baseline...")
    else:
        print("\nSkipping classical solver (--skip-classical)")
    
    # Setup sampler
    sampler = None
    if args.backend == 'hardware':
        if not HAS_IBM_RUNTIME:
            print("\nERROR: qiskit-ibm-runtime not installed")
            print("Install with: pip install qiskit-ibm-runtime")
            sys.exit(1)
        
        print("\nConnecting to IBM Quantum...")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            token = os.getenv('IBM_QUANTUM_API_TOKEN')
            if not token:
                print("ERROR: IBM_QUANTUM_API_TOKEN not found in .env file")
                print("Get token from: https://quantum.ibm.com/account")
                sys.exit(1)
            
            service = QiskitRuntimeService(
                channel='ibm_quantum_platform',
                token=token
            )
            backend = service.least_busy(
                min_num_qubits=graph.number_of_nodes(),
                operational=True,
                simulator=False
            )
            print(f"  Using backend: {backend.name}")
            sampler = RuntimeSampler(mode=backend)
        except Exception as e:
            print(f"ERROR connecting to IBM Quantum: {e}")
            sys.exit(1)
    else:
        print("\nUsing statevector simulator")
        sampler = StatevectorSampler()
    
    # Run QAOA
    print("\n" + "-"*70)
    qaoa_result = run_qaoa(
        graph=graph,
        p=args.p,
        shots=args.shots,
        max_iter=args.max_iter,
        method=args.method,
        sampler=sampler,
        verbose=True
    )
    
    # Analyze results
    print("="*70)
    print("RESULTS")
    print("="*70)
    
    best_solution = qaoa_result['top_solutions'][0]
    best_bitstring, best_cut_value, best_count = best_solution
    
    if classical_value is not None:
        approx_ratio = best_cut_value / classical_value
        print(f"\nClassical Optimal: {classical_value} cuts")
        print(f"QAOA Best:         {best_cut_value} cuts")
        print(f"Approximation ratio: {approx_ratio:.3f}")
    else:
        print(f"\nQAOA Best: {best_cut_value} cuts")
    
    print(f"\nTop 5 solutions:")
    for i, (bitstring, cut_val, count) in enumerate(qaoa_result['top_solutions'][:5], 1):
        prob = count / sum(qaoa_result['final_counts'].values())
        print(f"  {i}. Cut value: {cut_val:2d}, Probability: {prob:.3f}, Bitstring: {bitstring}")
    
    # Plot convergence
    if not args.no_plot:
        plt.figure(figsize=PLOT_CONFIG['figsize'])
        plt.plot(qaoa_result['energy_history'], 'b-', linewidth=2, label='QAOA')
        
        if classical_value is not None:
            plt.axhline(
                y=-classical_value/2,
                color='r',
                linestyle='--',
                label=f'Classical optimal (E={-classical_value/2:.2f})'
            )
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Energy', fontsize=12)
        plt.title(f'QAOA Convergence (p={args.p}, {args.nodes} nodes)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if PLOT_CONFIG.get('save_plots', False):
            plot_dir = PLOT_CONFIG.get('plot_dir', 'plots/')
            os.makedirs(plot_dir, exist_ok=True)
            filename = f"{plot_dir}qaoa_convergence_p{args.p}_n{args.nodes}.png"
            plt.savefig(filename, dpi=PLOT_CONFIG['dpi'])
            print(f"\nPlot saved to: {filename}")
        
        plt.show()
    
    # Save results if requested
    if args.save_results:
        import json
        results_data = {
            'graph_type': args.graph_type,
            'num_nodes': args.nodes if args.graph_type != 'grid' else args.rows * args.cols,
            'num_edges': num_edges,
            'p': args.p,
            'shots': args.shots,
            'optimizer': args.method,
            'classical_optimal': classical_value,
            'qaoa_best': best_cut_value,
            'approximation_ratio': best_cut_value / classical_value if classical_value else None,
            'optimal_energy': qaoa_result['optimal_energy'],
            'num_iterations': qaoa_result['num_iterations'],
            'converged': qaoa_result['success'],
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")
    
    print("\n" + "="*70)
    print("QAOA Execution Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
