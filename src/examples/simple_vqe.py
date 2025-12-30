"""
Simple example demonstrating basic VQE usage.

This script shows how to use the VQE implementation for a simple
2-qubit Heisenberg model calculation.
"""

import sys
sys.path.append('..')

from scripts.ansatz import he_ansatz_2q
from scripts.hamiltonian import heisenberg_xxz_2q
from scripts.energy_evaluation import energy_expectation
from scripts.optimization import run_vqe


def main():
    """Run a simple VQE example."""
    
    print("=" * 60)
    print("Simple VQE Example: 2-Qubit Heisenberg Model")
    print("=" * 60)
    
    # Step 1: Create the ansatz (parameterized quantum circuit)
    print("\n1. Creating ansatz...")
    ansatz, theta = he_ansatz_2q(depth=1)
    print(f"   ✓ Ansatz with {ansatz.num_parameters} parameters created")
    print(f"   ✓ Circuit depth: {ansatz.depth()}")
    
    # Step 2: Define the Hamiltonian
    print("\n2. Defining Hamiltonian...")
    J = 1.0      # Coupling strength
    delta = 1.0  # Anisotropy (Δ=1 gives isotropic XXX model)
    H = heisenberg_xxz_2q(J=J, delta=delta)
    print(f"   ✓ Heisenberg XXZ with J={J}, Δ={delta}")
    print(f"   ✓ Hamiltonian terms: {H.paulis}")
    
    # Step 3: Run VQE optimization
    print("\n3. Running VQE optimization...")
    print("   (This may take a minute...)\n")
    
    result = run_vqe(
        ansatz=ansatz,
        hamiltonian=H,
        energy_func=energy_expectation,  # Use exact statevector
        method="COBYLA",                  # Derivative-free optimizer
        max_iter=100                      # Maximum iterations
    )
    
    # Step 4: Display results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Ground state energy: {result['optimal_energy']:.8f}")
    print(f"True ground state:   -3.00000000 (for J=1, Δ=1)")
    print(f"Error:               {abs(result['optimal_energy'] + 3.0):.8f}")
    print(f"Iterations:          {result['num_iterations']}")
    print(f"Success:             {result['success']}")
    print(f"\nOptimal parameters:")
    for i, param in enumerate(result['optimal_params']):
        print(f"   θ[{i}] = {param:.6f}")
    
    print("\n" + "=" * 60)
    print("✅ Example complete!")
    print("=" * 60)
    
    # Physical interpretation
    print("\n💡 Physical Interpretation:")
    print("   The ground state of the isotropic Heisenberg model (Δ=1)")
    print("   is the singlet state: |ψ⟩ = (|01⟩ - |10⟩)/√2")
    print("   This state has maximum entanglement (spin-0 total)")
    print("   and energy E₀ = -3J = -3.0")


if __name__ == "__main__":
    main()
