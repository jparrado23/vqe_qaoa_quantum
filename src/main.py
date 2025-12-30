"""
Main execution script for VQE on Heisenberg model.

This script demonstrates the complete VQE workflow:
1. Exact statevector simulation
2. Shot-based simulation
3. Real quantum hardware execution (optional)

Usage:
    python main.py [--hardware] [--shots N] [--max-iter N]
"""

import argparse
import os
import numpy as np
from dotenv import load_dotenv
from scripts.ansatz import he_ansatz_2q
from scripts.hamiltonian import heisenberg_xxz_2q
from scripts.energy_evaluation import (
    energy_expectation,
    energy_expectation_shots,
    energy_expectation_hardware
)
from scripts.optimization import run_vqe
from utils.visualization import plot_comparison, print_vqe_summary
from utils.transpilation import transpile_for_hardware, transpile_hamiltonian
from config.settings import (
    ANSATZ_CONFIG,
    HAMILTONIAN_CONFIG,
    OPTIMIZER_CONFIG,
    SIMULATION_CONFIG,
    HARDWARE_CONFIG
)

# Load environment variables from .env file
load_dotenv()


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run VQE for Heisenberg model')
    parser.add_argument(
        '--hardware',
        action='store_true',
        help='Run on real quantum hardware (requires IBM Quantum account)'
    )
    parser.add_argument(
        '--shots',
        type=int,
        default=SIMULATION_CONFIG['shots'],
        help=f'Number of shots for simulation (default: {SIMULATION_CONFIG["shots"]})'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=OPTIMIZER_CONFIG['max_iter'],
        help=f'Maximum optimization iterations (default: {OPTIMIZER_CONFIG["max_iter"]})'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=ANSATZ_CONFIG['depth'],
        help=f'Ansatz depth (default: {ANSATZ_CONFIG["depth"]})'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("VQE for 2-Qubit Heisenberg XXZ Model")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: Setup - Create ansatz and Hamiltonian
    # ========================================================================
    print("\n📋 Setting up problem...")
    
    ansatz, theta = he_ansatz_2q(depth=args.depth)
    print(f"✓ Ansatz: {ansatz.num_parameters} parameters, depth={ansatz.depth()}")
    
    H = heisenberg_xxz_2q(
        J=HAMILTONIAN_CONFIG['J'],
        delta=HAMILTONIAN_CONFIG['delta']
    )
    print(f"✓ Hamiltonian: J={HAMILTONIAN_CONFIG['J']}, Δ={HAMILTONIAN_CONFIG['delta']}")
    print(f"  Terms: {H.paulis}")
    
    results = {}
    
    # ========================================================================
    # STEP 2: Exact statevector simulation
    # ========================================================================
    print("\n" + "=" * 70)
    print("1️⃣  EXACT STATEVECTOR SIMULATION")
    print("=" * 70)
    
    vqe_result_exact = run_vqe(
        ansatz=ansatz,
        hamiltonian=H,
        energy_func=energy_expectation,
        method=OPTIMIZER_CONFIG['method'],
        max_iter=args.max_iter
    )
    results['Exact'] = vqe_result_exact
    print_vqe_summary(vqe_result_exact, "Exact Statevector")
    
    # ========================================================================
    # STEP 3: Shot-based simulation
    # ========================================================================
    print("\n" + "=" * 70)
    print("2️⃣  SHOT-BASED SIMULATION")
    print("=" * 70)
    
    vqe_result_shots = run_vqe(
        ansatz=ansatz,
        hamiltonian=H,
        energy_func=lambda c, h, p: energy_expectation_shots(c, h, p, shots=args.shots),
        initial_params=None,  # Random initialization
        method=OPTIMIZER_CONFIG['method'],
        max_iter=args.max_iter
    )
    results['Shots'] = vqe_result_shots
    print_vqe_summary(vqe_result_shots, f"Shot-based ({args.shots} shots)")
    
    # ========================================================================
    # STEP 4: Real quantum hardware (optional)
    # ========================================================================
    if args.hardware:
        print("\n" + "=" * 70)
        print("3️⃣  REAL QUANTUM HARDWARE")
        print("=" * 70)
        
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            # Load API token from environment variable
            IBM_API_TOKEN = os.getenv('IBM_QUANTUM_API_TOKEN')
            
            if not IBM_API_TOKEN:
                raise ValueError(
                    "IBM_QUANTUM_API_TOKEN not found in environment variables.\n"
                    "Please create a .env file in the project root with your API token:\n"
                    "IBM_QUANTUM_API_TOKEN=your_token_here\n"
                    "Get your token from: https://quantum.ibm.com/account"
                )
            
            # Connect to IBM Quantum
            print("Connecting to IBM Quantum...")
            
            # Save credentials (only needed once)
            try:
                QiskitRuntimeService.save_account(
                    channel="ibm_quantum_platform",
                    token=IBM_API_TOKEN,
                    overwrite=True
                )
            except Exception as e:
                print(f"Note: Could not save account (might already exist): {e}")
            
            # Load service
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
            backend = service.least_busy(simulator=False, operational=True, min_num_qubits=2)
            
            print(f"✓ Backend: {backend.name}")
            print(f"  Qubits: {backend.num_qubits}")
            print(f"  Pending jobs: {backend.status().pending_jobs}")
            
            # Transpile for hardware
            print("\nTranspiling for hardware...")
            transpiled_ansatz, info = transpile_for_hardware(ansatz, backend)
            print(f"✓ Physical qubits: {info['physical_qubits']}")
            print(f"  Circuit depth: {info['original_depth']} → {info['transpiled_depth']}")
            
            # Transpile Hamiltonian
            H_transpiled = transpile_hamiltonian(
                H,
                info['physical_qubits'],
                backend.num_qubits
            )
            print(f"✓ Hamiltonian expanded to {backend.num_qubits} qubits")
            
            # Run VQE on hardware
            print("\n⚠️  Starting hardware execution...")
            print(f"Using warm start from simulation: {HARDWARE_CONFIG['use_warm_start']}")
            
            initial_params_hw = (
                vqe_result_exact['optimal_params']
                if HARDWARE_CONFIG['use_warm_start']
                else None
            )
            
            vqe_result_hardware = run_vqe(
                ansatz=transpiled_ansatz,
                hamiltonian=H_transpiled,
                energy_func=lambda c, h, p: energy_expectation_hardware(
                    c, h, p,
                    backend=backend,
                    shots=HARDWARE_CONFIG['shots']
                ),
                initial_params=initial_params_hw,
                method=OPTIMIZER_CONFIG['method'],
                max_iter=HARDWARE_CONFIG['max_iter_hardware']
            )
            results['Hardware'] = vqe_result_hardware
            print_vqe_summary(vqe_result_hardware, f"Hardware ({backend.name})")
            
        except Exception as e:
            print(f"❌ Hardware execution failed: {e}")
            print("Continuing with simulation results only...")
    
    # ========================================================================
    # STEP 5: Analysis and comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE COMPARISON")
    print("=" * 70)
    
    # Print comparison table
    reference_energy = results['Exact']['optimal_energy']
    print(f"\n{'Method':<20} {'Energy':>12} {'Error':>12} {'Iterations':>12}")
    print("-" * 70)
    
    for method_name, result in results.items():
        energy = result['optimal_energy']
        error = abs(energy - reference_energy)
        iters = result['num_iterations']
        print(f"{method_name:<20} {energy:>+12.6f} {error:>12.6f} {iters:>12}")
    
    print("\n💡 Insights:")
    if 'Shots' in results:
        shot_error = abs(results['Shots']['optimal_energy'] - reference_energy)
        shot_pct = (shot_error / abs(reference_energy)) * 100
        print(f"   • Shot noise ({args.shots} shots): ~{shot_pct:.2f}% error")
    
    if 'Hardware' in results:
        hw_error = abs(results['Hardware']['optimal_energy'] - reference_energy)
        hw_pct = (hw_error / abs(reference_energy)) * 100
        print(f"   • Hardware noise: ~{hw_pct:.2f}% error")
        
        if 'Shots' in results:
            if hw_error > shot_error:
                print("   • Hardware is noisier than shot simulation (expected)")
            else:
                print("   • Hardware performed better than expected!")
    
    # Generate comparison plot
    try:
        import matplotlib.pyplot as plt
        print("\n📈 Generating comparison plots...")
        fig = plot_comparison(results, reference_key='Exact')
        plt.savefig('vqe_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Plot saved: vqe_comparison.png")
        plt.show()
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
    
    print("\n" + "=" * 70)
    print("✅ VQE execution complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
