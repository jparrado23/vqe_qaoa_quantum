"""
Configuration module for VQE experiments.

This module centralizes all configuration parameters for reproducibility
and easy experimentation.
"""

# Hardware configuration (IBM Quantum)
IBM_QUANTUM_CONFIG = {
    'channel': 'ibm_quantum_platform',
    'default_shots': 1024,
    'resilience_level': 1,
    'optimization_level': 3
}

# VQE optimizer settings
OPTIMIZER_CONFIG = {
    'method': 'COBYLA',
    'max_iter': 200,
    'tol': 1e-6
}

# Ansatz configuration
ANSATZ_CONFIG = {
    'depth': 1,
    'num_qubits': 2
}

# Hamiltonian parameters (Heisenberg XXZ)
HAMILTONIAN_CONFIG = {
    'J': 1.0,
    'delta': 1.0
}

# Shot-based simulation settings
SIMULATION_CONFIG = {
    'shots': 2000
}

# Hardware execution settings
HARDWARE_CONFIG = {
    'shots': 1024,
    'max_iter_hardware': 20,  # Conservative for queue times
    'use_warm_start': True    # Use simulation results as initial guess
}

# Visualization settings
PLOT_CONFIG = {
    'figsize': (10, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid'
}
