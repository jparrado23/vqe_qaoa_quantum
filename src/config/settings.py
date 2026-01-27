"""
Configuration module for VQE and QAOA experiments.

This module centralizes all configuration parameters for reproducibility
and easy experimentation across different quantum algorithms.
"""

# ============================================================================
# ALGORITHM SELECTION
# ============================================================================

ALGORITHM = {
    'type': 'VQE',  # Options: 'VQE', 'QAOA'
}

# ============================================================================
# HARDWARE CONFIGURATION (IBM Quantum)
# ============================================================================

IBM_QUANTUM_CONFIG = {
    'channel': 'ibm_quantum_platform',
    'default_shots': 1024,
    'resilience_level': 1,
    'optimization_level': 3
}

# ============================================================================
# VQE CONFIGURATION
# ============================================================================

# VQE optimizer settings
OPTIMIZER_CONFIG = {
    'method': 'COBYLA',
    'max_iter': 200,
    'tol': 1e-6
}

# Ansatz configuration for VQE
ANSATZ_CONFIG = {
    'depth': 1,
    'num_qubits': 2
}

# Hamiltonian parameters (Heisenberg XXZ)
HAMILTONIAN_CONFIG = {
    'J': 1.0,
    'delta': 1.0
}

# Shot-based simulation settings for VQE
SIMULATION_CONFIG = {
    'shots': 2000
}

# Hardware execution settings for VQE
HARDWARE_CONFIG = {
    'shots': 1024,
    'max_iter_hardware': 20,  # Conservative for queue times
    'use_warm_start': True    # Use simulation results as initial guess
}

# ============================================================================
# QAOA CONFIGURATION
# ============================================================================

# QAOA circuit parameters
QAOA_CIRCUIT_CONFIG = {
    'p': 5,  # Number of QAOA layers
    'initial_state': 'superposition',  # Options: 'superposition', 'custom'
    'mixer': 'x_mixer',  # Options: 'x_mixer', 'xy_mixer', 'custom'
}

# QAOA optimizer settings
QAOA_OPTIMIZER_CONFIG = {
    'method': 'COBYLA',  # Options: 'COBYLA', 'SLSQP', 'Powell', 'L-BFGS-B'
    'max_iter': 200,
    'tol': 1e-6,
    'initial_params': 'random',  # Options: 'random', 'heuristic', 'custom'
}

# Graph/Problem configuration for QAOA
QAOA_PROBLEM_CONFIG = {
    'problem_type': 'maxcut',  # Options: 'maxcut', 'max_independent_set'
    'graph_type': 'random',  # Options: 'random', 'regular', 'grid', 'hardware_topology'
    'num_nodes': 12,
    'edge_probability': 0.5,  # For random graphs
    'degree': 3,  # For regular graphs
    'seed': 42,  # For reproducibility
}

# QAOA hardware execution
QAOA_HARDWARE_CONFIG = {
    'shots': 1024,
    'use_transpilation': True,
    'use_initial_layout': True,  # Constrain qubit mapping
    'dynamical_decoupling': True,
    'dd_sequence': 'XY4',
    'twirling_gates': True,
    'twirling_measure': True,
}

# QAOA CVaR settings (error mitigation)
QAOA_CVAR_CONFIG = {
    'use_cvar': False,  # Enable CVaR cost function
    'alpha': 0.5,  # CVaR confidence level (0 < alpha <= 1)
    'adaptive_alpha': True,  # Adapt alpha based on circuit error rate
}

# Classical comparison
QAOA_CLASSICAL_CONFIG = {
    'solve_classical': True,  # Solve problem classically for comparison
    'max_classical_nodes': 100,  # Maximum graph size for exact classical solver
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

PLOT_CONFIG = {
    'figsize': (10, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'save_plots': True,
    'plot_dir': 'plots/'
}
