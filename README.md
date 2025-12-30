# VQE & QAOA Quantum Algorithms

Implementation of Variational Quantum Eigensolver (VQE) and Quantum Approximate Optimization Algorithm (QAOA) for solving quantum many-body problems and combinatorial optimization.

## Overview

This repository contains a complete implementation of VQE for the 2-qubit Heisenberg XXZ model, demonstrating the full workflow from exact simulation to real quantum hardware execution on IBM Quantum devices.

## Features

- **Modular Architecture**: Clean separation between ansatz, Hamiltonian, energy evaluation, and optimization
- **Multiple Backends**: Exact statevector, shot-based simulation, and real quantum hardware
- **Hardware Integration**: Full IBM Quantum support with automatic transpilation and error mitigation
- **Visualization Tools**: Energy convergence plots and method comparison charts
- **CLI Interface**: Command-line tool for running experiments
- **Well-Documented**: Comprehensive docstrings and usage examples

## Project Structure

```
vqe_qaoa_quantum/
├── src/
│   ├── config/              # Configuration settings
│   ├── scripts/             # Core VQE implementation
│   │   ├── ansatz.py        # Quantum circuit construction
│   │   ├── hamiltonian.py   # Hamiltonian definitions
│   │   ├── energy_evaluation.py  # Energy evaluation backends
│   │   └── optimization.py  # VQE optimization loop
│   ├── utils/               # Utility functions
│   │   ├── transpilation.py     # Hardware transpilation
│   │   └── visualization.py     # Plotting tools
│   ├── examples/            # Example scripts
│   └── main.py             # Main execution script
│
├── notebooks/               # Jupyter notebooks
│   └── vqe_heisenberg_2qubit.ipynb
│
└── requirements.txt         # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10 or higher
- IBM Quantum account (for hardware execution)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jparrado23/vqe_qaoa_quantum.git
cd vqe_qaoa_quantum
```

2. Create a virtual environment:
```bash
conda create -n quantum_py312 python=3.12
conda activate quantum_py312
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure IBM Quantum credentials (for hardware execution):
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add the IBM Quantum API token
# Token available at: https://quantum.ibm.com/account
# Replace 'your_ibm_quantum_api_token_here' with the actual token
```

The `.env` file should contain:
```bash
IBM_QUANTUM_API_TOKEN=your_actual_token_here
```

**Security Note**: The `.env` file is already in `.gitignore` and will not be committed to version control.

## Quick Start

### Run Basic Example

```bash
cd src/examples
python simple_vqe.py
```

### Run Full VQE Workflow

```bash
cd src
python main.py
```

This executes:
1. Exact statevector simulation (validation)
2. Shot-based simulation (realistic noise)
3. Comprehensive comparison and analysis

### Run on Real Quantum Hardware

```bash
python main.py --hardware
```

**Note**: Requires IBM Quantum API token configured in `.env` file (see Installation step 4).

### Command Line Options

```bash
# Custom shot count
python main.py --shots 5000

# More optimization iterations
python main.py --max-iter 300

# Deeper ansatz circuit
python main.py --depth 2

# Combine options
python main.py --shots 5000 --max-iter 300 --depth 2
```

## Using as a Library

```python
from src.scripts.ansatz import he_ansatz_2q
from src.scripts.hamiltonian import heisenberg_xxz_2q
from src.scripts.energy_evaluation import energy_expectation
from src.scripts.optimization import run_vqe

# Setup problem
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
print(f"Optimal parameters: {result['optimal_params']}")
```

## Physics Background

### Heisenberg XXZ Model

The Hamiltonian describes nearest-neighbor spin-1/2 interactions:

```
H = J(σₓ⊗σₓ + σᵧ⊗σᵧ + Δ·σᵤ⊗σᵤ)
```

- **J > 0**: Antiferromagnetic coupling (anti-aligned spins favored)
- **J < 0**: Ferromagnetic coupling (aligned spins favored)
- **Δ = 1**: Isotropic (XXX) model with full rotational symmetry
- **Δ ≠ 1**: Anisotropic model with broken symmetry along z-axis

For J=1, Δ=1, the ground state is the singlet state with energy E₀ = -3.

### VQE Algorithm

1. Prepare parameterized quantum state |ψ(θ)⟩
2. Measure energy expectation value E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
3. Use classical optimizer to minimize E(θ)
4. Iterate until convergence

By the variational principle, E(θ) ≥ E₀, so minimization approximates the ground state.

## Configuration

Customize parameters in `src/config/settings.py`:

```python
# Hamiltonian parameters
HAMILTONIAN_CONFIG = {
    'J': 1.0,      # Coupling strength
    'delta': 1.0   # Anisotropy parameter
}

# Optimizer settings
OPTIMIZER_CONFIG = {
    'method': 'COBYLA',
    'max_iter': 200
}

# Simulation settings
SIMULATION_CONFIG = {
    'shots': 2000
}
```

## Examples

### Simple VQE Example
Demonstrates basic VQE workflow with exact simulation:
```bash
cd src/examples
python simple_vqe.py
```

### Shot Comparison Example
Compares exact vs shot-based simulation with different shot counts:
```bash
cd src/examples
python compare_shots.py
```

## Hardware Execution

### IBM Quantum Setup

1. Create an IBM Quantum account at https://quantum.ibm.com/
2. Obtain the API token from the account page
3. Configure the token:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_API_TOKEN_HERE",
    overwrite=True
)
```

### Hardware Considerations

- **Transpilation**: Circuits are automatically transpiled to match hardware topology
- **Error Mitigation**: Resilience level 1 applies basic noise reduction
- **Queue Times**: Jobs may wait minutes to hours depending on system availability
- **Warm Start**: Uses simulation results as initial guess to reduce QPU time

## Documentation

- **Quick Start**: See `QUICKSTART.md` for detailed usage guide
- **Source Documentation**: See `src/README.md` for complete API reference
- **Notebooks**: Interactive tutorials in `notebooks/` directory

## Dependencies

- qiskit >= 2.0.0
- qiskit-ibm-runtime >= 0.30.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0

## Results

The implementation demonstrates:
- Ground state energy convergence within ~100 iterations
- Shot noise impact on optimization (typically <1% error with 2000 shots)
- Hardware execution with realistic noise (varies by device quality)
- Successful validation against known analytical results

## Contributing

Contributions are welcome! Areas for extension:
- Additional Hamiltonians (Ising, molecular systems)
- Alternative ansätze (UCCSD, problem-inspired)
- Different optimizers (SPSA, Adam, Nelder-Mead)
- Multi-qubit systems
- QAOA implementation

## References

- Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor" (2014)
- IBM Quantum Documentation: https://quantum.ibm.com/
- Qiskit Documentation: https://docs.quantum.ibm.com/

## License

MIT License

## Author

Juan Parrado

## Acknowledgments

- IBM Quantum for providing access to quantum hardware
- Qiskit development team for the quantum computing framework
