"""
Example demonstrating shot-based simulation vs exact evaluation.

This script compares VQE results using:
1. Exact statevector simulation (no noise)
2. Shot-based simulation (finite sampling noise)
"""

import sys
sys.path.append('..')

import numpy as np
from scripts.ansatz import he_ansatz_2q
from scripts.hamiltonian import heisenberg_xxz_2q
from scripts.energy_evaluation import energy_expectation, energy_expectation_shots
from scripts.optimization import run_vqe


def main():
    """Compare exact vs shot-based VQE."""
    
    print("=" * 70)
    print("VQE Comparison: Exact vs Shot-based Simulation")
    print("=" * 70)
    
    # Setup
    ansatz, theta = he_ansatz_2q(depth=1)
    H = heisenberg_xxz_2q(J=1.0, delta=1.0)
    
    print(f"\nProblem setup:")
    print(f"  Ansatz: {ansatz.num_parameters} parameters")
    print(f"  Hamiltonian: Heisenberg XXZ (J=1.0, Δ=1.0)")
    
    # Run exact VQE
    print("\n" + "=" * 70)
    print("1. Exact Statevector Simulation")
    print("=" * 70)
    
    result_exact = run_vqe(
        ansatz=ansatz,
        hamiltonian=H,
        energy_func=energy_expectation,
        method="COBYLA",
        max_iter=150
    )
    
    # Run shot-based VQE with different shot counts
    shot_counts = [500, 2000, 5000]
    results_shots = {}
    
    for shots in shot_counts:
        print("\n" + "=" * 70)
        print(f"2. Shot-based Simulation ({shots} shots)")
        print("=" * 70)
        
        result = run_vqe(
            ansatz=ansatz,
            hamiltonian=H,
            energy_func=lambda c, h, p: energy_expectation_shots(c, h, p, shots=shots),
            initial_params=None,  # Random start
            method="COBYLA",
            max_iter=150
        )
        results_shots[shots] = result
    
    # Comparison
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    
    exact_energy = result_exact['optimal_energy']
    
    print(f"\n{'Method':<30} {'Energy':>12} {'Error':>12} {'Iterations':>10}")
    print("-" * 70)
    print(f"{'Exact (reference)':<30} {exact_energy:>+12.6f} {'0.000000':>12} {result_exact['num_iterations']:>10}")
    
    for shots, result in results_shots.items():
        energy = result['optimal_energy']
        error = abs(energy - exact_energy)
        iters = result['num_iterations']
        print(f"{f'Shots ({shots})':<30} {energy:>+12.6f} {error:>12.6f} {iters:>10}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("💡 Analysis")
    print("=" * 70)
    
    print("\nShot noise vs accuracy trade-off:")
    for shots, result in results_shots.items():
        error = abs(result['optimal_energy'] - exact_energy)
        error_pct = (error / abs(exact_energy)) * 100
        print(f"  {shots:5d} shots → {error:.6f} error (~{error_pct:.2f}%)")
    
    print("\nKey insights:")
    print("  • More shots = better accuracy but slower execution")
    print("  • Shot noise introduces stochasticity in optimization")
    print("  • May need more iterations to converge with fewer shots")
    print("  • 2000-5000 shots typical for good accuracy/speed balance")
    
    print("\n" + "=" * 70)
    print("✅ Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
