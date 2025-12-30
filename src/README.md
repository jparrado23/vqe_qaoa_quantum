# VQE for Heisenberg XXZ Model

This package implements the Variational Quantum Eigensolver (VQE) algorithm for finding the ground state energy of the 2-qubit Heisenberg XXZ model.

## Project Structure

```
src/
├── __init__.py           # Package initialization
├── main.py               # Main execution script
├── config/               # Configuration settings
│   ├── __init__.py
│   └── settings.py       # Default parameters
├── scripts/              # Core VQE implementation
│   ├── __init__.py
│   ├── ansatz.py         # Quantum circuit ansatz construction
│   ├── hamiltonian.py    # Hamiltonian definition
│   ├── energy_evaluation.py  # Energy evaluation (exact/shots/hardware)
│   └── optimization.py   # VQE optimization loop
└── utils/                # Utility functions
    ├── __init__.py
    ├── transpilation.py  # Hardware transpilation utilities
    └── visualization.py  # Plotting and analysis tools
```

## Installation

1. **Create a virtual environment** (recommended):
```bash
conda create -n quantum_py312 python=3.12
conda activate quantum_py312
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Simulation Only)

Run VQE with exact statevector and shot-based simulation:

```bash
python src/main.py
```

### Custom Parameters

```bash
# Use more shots for better accuracy
python src/main.py --shots 5000

# Use more optimization iterations
python src/main.py --max-iter 300

# Use deeper ansatz
python src/main.py --depth 2
```

### Real Quantum Hardware

To run on IBM Quantum hardware:

```bash
python src/main.py --hardware
```

**Note**: Requires IBM Quantum account and API token configured.

## Python API Usage

Modules can be imported and used directly:

```python
from src.scripts.ansatz import he_ansatz_2q
from src.scripts.hamiltonian import heisenberg_xxz_2q
from src.scripts.energy_evaluation import energy_expectation
from src.scripts.optimization import run_vqe

# Create ansatz and Hamiltonian
ansatz, theta = he_ansatz_2q(depth=1)
H = heisenberg_xxz_2q(J=1.0, delta=1.0)

# Run VQE
result = run_vqe(
    ansatz=ansatz,
    hamiltonian=H,
    energy_func=energy_expectation,
    method="COBYLA",
    max_iter=200
)

print(f"Ground state energy: {result['optimal_energy']:.6f}")
```

## Features

### Three Evaluation Modes

1. **Exact Statevector** (`energy_expectation`)
   - Noiseless simulation
   - Perfect accuracy
   - Fast for small systems
   - Ideal for validation

2. **Shot-based Simulation** (`energy_expectation_shots`)
   - Mimics real hardware
   - Finite sampling noise
   - Configurable shot count
   - Realistic performance

3. **Real Hardware** (`energy_expectation_hardware`)
   - IBM Quantum execution
   - Physical noise (gates, decoherence, readout)
   - Error mitigation enabled
   - Queue-based execution

### Key Components

- **Hardware-Efficient Ansatz**: Parameterized quantum circuit optimized for NISQ devices
- **Heisenberg XXZ Hamiltonian**: 2-qubit spin system with tunable anisotropy
- **Classical Optimization**: COBYLA optimizer (derivative-free, robust to noise)
- **Transpilation**: Automatic circuit and Hamiltonian mapping for hardware
- **Visualization**: Energy convergence plots and method comparisons

## Configuration

Edit `src/config/settings.py` to customize:

- Hamiltonian parameters (J, Δ)
- Ansatz depth
- Optimizer settings
- Shot counts
- Hardware execution parameters

## Example Output

```
======================================================================
VQE for 2-Qubit Heisenberg XXZ Model
======================================================================

📋 Setting up problem...
✓ Ansatz: 4 parameters, depth=1
✓ Hamiltonian: J=1.0, Δ=1.0
  Terms: ['XX', 'YY', 'ZZ']

======================================================================
1️⃣  EXACT STATEVECTOR SIMULATION
======================================================================
Starting VQE optimization...
Optimizer: COBYLA, Max iterations: 200
Ansatz: 4 parameters, depth=1
Initial energy: +1.234567
--------------------------------------------------
Iteration   1: E = +1.234567
Iteration  20: E = -2.876543
...
--------------------------------------------------
Optimization complete!
Final energy: -3.000000
Total iterations: 87

======================================================================
📊 COMPREHENSIVE COMPARISON
======================================================================

Method               Energy       Error   Iterations
----------------------------------------------------------------------
Exact                -3.000000    0.000000           87
Shots                -2.987654    0.012346          103
Hardware             -2.945321    0.054679           18

💡 Insights:
   • Shot noise (2000 shots): ~0.41% error
   • Hardware noise: ~1.82% error
   • Hardware is noisier than shot simulation (expected)
```

## Physics Background

### Heisenberg XXZ Model

The Hamiltonian describes nearest-neighbor spin-1/2 interactions:

```
H = J(σₓ⊗σₓ + σᵧ⊗σᵧ + Δ·σᵤ⊗σᵤ)
```

- **J > 0**: Antiferromagnetic (anti-aligned spins favored)
- **J < 0**: Ferromagnetic (aligned spins favored)
- **Δ = 1**: Isotropic (XXX model, full rotational symmetry)
- **Δ ≠ 1**: Anisotropic (broken symmetry along z-axis)

For J=1, Δ=1, the ground state is the singlet:
```
|ψ₀⟩ = (|01⟩ - |10⟩)/√2
E₀ = -3
```

### VQE Algorithm

1. **Prepare** parameterized quantum state |ψ(θ)⟩
2. **Measure** energy E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
3. **Optimize** θ classically to minimize E(θ)
4. **Iterate** until convergence

By the variational principle: E(θ) ≥ E₀ for all θ

## Hardware Execution Notes

- **Transpilation**: Circuits and Hamiltonians must be mapped to physical qubits
- **Queue Times**: Jobs may wait minutes to hours depending on demand
- **Error Mitigation**: Resilience level 1 applies basic noise reduction
- **Cost**: Monitor IBM Quantum dashboard for runtime usage
- **Warm Start**: Use simulation results as initial guess to reduce QPU time

## References

- Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor" (2014)
- IBM Quantum Documentation: https://quantum.ibm.com/
- Qiskit Textbook: https://learn.qiskit.org/

## License

MIT License

## Author

Juan Parrado
