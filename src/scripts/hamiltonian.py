"""
Hamiltonian construction module.

Functions for constructing quantum Hamiltonians for many-body spin systems.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def heisenberg_xxz_2q(J: float = 1.0, delta: float = 1.0) -> SparsePauliOp:
    """
    Construct the XXZ Heisenberg Hamiltonian for 2 qubits.
    
    The Hamiltonian describes nearest-neighbor spin-1/2 interactions:
    
        H = J(σ_x ⊗ σ_x + σ_y ⊗ σ_y + Δ·σ_z ⊗ σ_z)
    
    where σ_x, σ_y, σ_z are Pauli matrices and ⊗ denotes tensor product.
    
    Physical interpretation:
    - J > 0: Antiferromagnetic coupling (favors anti-aligned spins → singlet state)
    - J < 0: Ferromagnetic coupling (favors aligned spins → triplet state)
    - Δ = 1: Isotropic (XXX) model with full rotational symmetry
    - Δ ≠ 1: Anisotropic model breaking symmetry along z-axis
    - Δ < 1: Favors alignment in xy-plane (easy-plane anisotropy)
    - Δ > 1: Favors alignment along z-axis (easy-axis anisotropy)

    Parameters
    ----------
    J : float, default=1.0
        Coupling strength. Positive for antiferromagnetic, negative for ferromagnetic.
    delta : float, default=1.0
        Anisotropy parameter. Controls the relative strength of ZZ interaction.
        
    Returns
    -------
    SparsePauliOp
        The Hamiltonian as a sum of weighted Pauli strings:
        - "XX" term with coefficient J
        - "YY" term with coefficient J
        - "ZZ" term with coefficient J*delta
        
    Examples
    --------
    >>> # Isotropic antiferromagnetic Heisenberg model
    >>> H_iso = heisenberg_xxz_2q(J=1.0, delta=1.0)
    
    >>> # Anisotropic model with stronger z-coupling
    >>> H_aniso = heisenberg_xxz_2q(J=1.0, delta=2.0)
    
    >>> # Ferromagnetic XXZ model
    >>> H_ferro = heisenberg_xxz_2q(J=-1.0, delta=1.0)
    
    Notes
    -----
    - Ground state for J=1, Δ=1 is the singlet |ψ⟩ = (|01⟩ - |10⟩)/√2 
      with energy E₀ = -3J
    - This Hamiltonian is small enough to solve exactly classically,
      making it ideal for VQE validation
    - For real hardware, each Pauli term requires separate measurement
      in the appropriate basis (X, Y, or Z)
    """
    # Define Pauli strings for the three interaction terms
    paulis = ["XX", "YY", "ZZ"]
    
    # Set coefficients: J for XX and YY, J*delta for ZZ
    coeffs = [J, J, J * delta]
    
    # Create SparsePauliOp (efficient representation for Pauli sums)
    return SparsePauliOp(paulis, coeffs=np.asarray(coeffs, dtype=complex))
