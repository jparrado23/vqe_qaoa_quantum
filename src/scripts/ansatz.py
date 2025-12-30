"""
Ansatz construction module.

Functions for constructing parameterized quantum circuits (ansätze)
for variational quantum algorithms.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def he_ansatz_2q(depth: int = 1):
    """
    Hardware-efficient ansatz for 2 qubits.
    
    This ansatz uses a simple structure that is native to most quantum hardware:
    - Single-qubit rotations (RY and RZ gates)
    - Entangling layers (CNOT gates)
    
    The ansatz is repeated for 'depth' layers, providing increased expressivity
    at the cost of circuit depth and noise susceptibility.

    Structure per layer:
      1. Single-qubit rotations: RY(θ) then RZ(θ) on each qubit
         (Together, RY and RZ can generate any single-qubit unitary)
      2. Entangling layer: CX(0 → 1) to create correlations between qubits

    Parameters
    ----------
    depth : int, default=1
        Number of ansatz layers. Each layer adds 4 parameters.
        
    Returns
    -------
    circuit : QuantumCircuit
        Parameterized quantum circuit (2 qubits)
    theta : ParameterVector
        Parameter vector containing all rotation angles (length = 4*depth)
        
    Raises
    ------
    ValueError
        If depth < 1
        
    Examples
    --------
    >>> ansatz, params = he_ansatz_2q(depth=2)
    >>> print(f"Circuit has {ansatz.num_parameters} parameters")
    Circuit has 8 parameters
    
    Notes
    -----
    - For depth=1: 4 parameters, suitable for shallow NISQ devices
    - For depth=2: 8 parameters, increased expressivity
    - Each additional layer adds 4 parameters and 5 gates
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    theta = ParameterVector("θ", length=4 * depth)
    qc = QuantumCircuit(2, name=f"HEA2q_d{depth}")

    k = 0
    for _ in range(depth):
        # Local single-qubit rotations on both qubits
        qc.ry(theta[k + 0], 0)  # RY rotation on qubit 0
        qc.rz(theta[k + 1], 0)  # RZ rotation on qubit 0
        qc.ry(theta[k + 2], 1)  # RY rotation on qubit 1
        qc.rz(theta[k + 3], 1)  # RZ rotation on qubit 1
        k += 4
        
        # Entangling layer: CNOT from qubit 0 to qubit 1
        qc.cx(0, 1)

    return qc, theta
