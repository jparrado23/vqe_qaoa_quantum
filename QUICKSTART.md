# Quick Start Guide - VQE Package

## 📁 Complete Structure

The `src/` directory is organized as follows:

```
src/
├── __init__.py                    # Package root
├── main.py                        # Main execution script
├── README.md                      # Full documentation
│
├── config/                        # Configuration
│   ├── __init__.py
│   └── settings.py                # All parameters (Hamiltonian, optimizer, etc.)
│
├── scripts/                       # Core VQE implementation
│   ├── __init__.py
│   ├── ansatz.py                  # Quantum circuit construction
│   ├── hamiltonian.py             # Hamiltonian definition
│   ├── energy_evaluation.py      # Energy evaluation (exact/shots/hardware)
│   └── optimization.py            # VQE optimization loop
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── transpilation.py           # Hardware transpilation
│   └── visualization.py           # Plotting tools
│
└── examples/                      # Example scripts
    ├── __init__.py
    ├── simple_vqe.py              # Basic usage
    └── compare_shots.py           # Exact vs shots comparison
```

## 🚀 Quick Start

### 1. Run the Simple Example

```bash
cd src/examples
python simple_vqe.py
```

This demonstrates basic VQE usage with exact statevector simulation.

### 2. Run the Full Main Script

```bash
cd src
python main.py
```

This runs both exact and shot-based simulations with full analysis.

### 3. Compare Different Shot Counts

```bash
cd src/examples
python compare_shots.py
```

This shows the effect of shot noise on VQE results.

### 4. Run on Real Hardware

```bash
cd src
python main.py --hardware
```

**Requirements:**
- IBM Quantum account
- API token configured
- Sufficient runtime allocation

## 📚 Using as a Library

```python
# Add src to path
import sys
sys.path.append('path/to/src')

# Import modules
from scripts.ansatz import he_ansatz_2q
from scripts.hamiltonian import heisenberg_xxz_2q
from scripts.energy_evaluation import energy_expectation
from scripts.optimization import run_vqe

# Run VQE
ansatz, theta = he_ansatz_2q(depth=1)
H = heisenberg_xxz_2q(J=1.0, delta=1.0)

result = run_vqe(
    ansatz=ansatz,
    hamiltonian=H,
    energy_func=energy_expectation,
    max_iter=200
)

print(f"Ground state energy: {result['optimal_energy']:.6f}")
```

## ⚙️ Configuration

Edit `src/config/settings.py` to customize:

```python
# Hamiltonian parameters
HAMILTONIAN_CONFIG = {
    'J': 1.0,      # Coupling strength
    'delta': 1.0   # Anisotropy
}

# Optimizer settings
OPTIMIZER_CONFIG = {
    'method': 'COBYLA',  # or 'SLSQP', 'Powell', etc.
    'max_iter': 200,
    'tol': 1e-6
}

# Shot-based simulation
SIMULATION_CONFIG = {
    'shots': 2000  # More shots = better accuracy
}
```

## 🔧 Command Line Options

```bash
# Custom shot count
python main.py --shots 5000

# More iterations
python main.py --max-iter 300

# Deeper ansatz
python main.py --depth 2

# Hardware execution
python main.py --hardware

# Combine options
python main.py --shots 5000 --max-iter 300 --depth 2
```

## 📊 Key Modules

### 1. Ansatz (`scripts/ansatz.py`)
```python
ansatz, theta = he_ansatz_2q(depth=1)
# Returns: QuantumCircuit and ParameterVector
```

### 2. Hamiltonian (`scripts/hamiltonian.py`)
```python
H = heisenberg_xxz_2q(J=1.0, delta=1.0)
# Returns: SparsePauliOp
```

### 3. Energy Evaluation (`scripts/energy_evaluation.py`)
```python
# Exact
energy = energy_expectation(circuit, hamiltonian, params)

# Shot-based
energy = energy_expectation_shots(circuit, hamiltonian, params, shots=2000)

# Hardware
energy = energy_expectation_hardware(circuit, hamiltonian, params, backend, shots=1024)
```

### 4. Optimization (`scripts/optimization.py`)
```python
result = run_vqe(
    ansatz=ansatz,
    hamiltonian=H,
    energy_func=energy_expectation,
    initial_params=None,
    method="COBYLA",
    max_iter=200
)
# Returns: dict with optimal_params, optimal_energy, num_iterations, success, energy_history
```

### 5. Transpilation (`utils/transpilation.py`)
```python
# Transpile circuit
transpiled_circuit, info = transpile_for_hardware(circuit, backend)

# Transpile Hamiltonian
H_transpiled = transpile_hamiltonian(H, physical_qubits, num_qubits_hardware)
```

### 6. Visualization (`utils/visualization.py`)
```python
# Plot energy history
fig = plot_energy_history(result['energy_history'], ground_state_energy=-3.0)

# Compare methods
fig = plot_comparison(results_dict, reference_key='exact')

# Print summary
print_vqe_summary(result, method_name="Exact Statevector")
```

## 🎯 Next Steps

1. **Explore examples**: Run all example scripts to understand usage patterns
2. **Modify parameters**: Edit `config/settings.py` to experiment with different settings
3. **Try different Hamiltonians**: Adjust J and Δ values to explore different physical systems
4. **Increase depth**: Test deeper ansätze (depth=2, 3, ...) for better expressivity
5. **Hardware execution**: Execute on real IBM Quantum hardware
6. **Visualization**: Generate plots and analyze convergence behavior

## 📖 Full Documentation

See `src/README.md` for complete documentation including:
- Detailed API reference
- Physics background
- Hardware execution notes
- References and resources

## ❓ Troubleshooting

**Import errors**: Verify the working directory or add `src` to the Python path.

**Hardware connection issues**: Confirm the IBM Quantum API token is configured in the `.env` file:
```bash
IBM_QUANTUM_API_TOKEN=your_token_here
```

**Slow convergence**: Consider:
- Increasing `max_iter`
- Providing better initial parameters
- Reducing shot noise (increase shots)
- Experimenting with different optimizers

## Getting Started

The VQE package is ready to use. All notebook functionality has been converted to modular, reusable Python scripts with proper documentation and examples.
