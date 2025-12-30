"""
Visualization utilities for VQE results.

Plotting functions for analyzing VQE optimization and comparing different
evaluation methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_energy_history(
    energy_history: List[float],
    title: str = "VQE Energy Convergence",
    ground_state_energy: Optional[float] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot VQE optimization energy history.
    
    Parameters
    ----------
    energy_history : list[float]
        Energy values at each iteration
    title : str, default="VQE Energy Convergence"
        Plot title
    ground_state_energy : float, optional
        True ground state energy (if known) to plot as reference
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
        
    Examples
    --------
    >>> result = run_vqe(ansatz, H, energy_expectation, max_iter=100)
    >>> fig = plot_energy_history(
    ...     result['energy_history'],
    ...     title="VQE Optimization (COBYLA)",
    ...     ground_state_energy=-3.0
    ... )
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = np.arange(1, len(energy_history) + 1)
    ax.plot(iterations, energy_history, 'b-', linewidth=2, label='VQE Energy')
    
    # Plot ground state reference if provided
    if ground_state_energy is not None:
        ax.axhline(
            y=ground_state_energy,
            color='r',
            linestyle='--',
            linewidth=2,
            label=f'True Ground State (E₀={ground_state_energy:.4f})'
        )
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add convergence info
    final_energy = energy_history[-1]
    if ground_state_energy is not None:
        error = abs(final_energy - ground_state_energy)
        ax.text(
            0.98, 0.02,
            f'Final: {final_energy:.6f}\nError: {error:.6f}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_comparison(
    results: Dict[str, Dict],
    reference_key: str = 'exact',
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare VQE results from different evaluation methods.
    
    Creates side-by-side plots showing:
    1. Energy comparison bar chart
    2. Convergence curves overlay
    
    Parameters
    ----------
    results : dict
        Dictionary mapping method names to VQE result dictionaries.
        Example: {
            'exact': vqe_result_exact,
            'shots': vqe_result_shots,
            'hardware': vqe_result_hardware
        }
    reference_key : str, default='exact'
        Which method to use as reference for error calculation
    figsize : tuple, default=(12, 5)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
        
    Examples
    --------
    >>> results = {
    ...     'Exact': vqe_result,
    ...     'Shots (2000)': vqe_result_shots,
    ...     'Hardware': vqe_result_hardware
    ... }
    >>> fig = plot_comparison(results, reference_key='Exact')
    >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract data
    methods = list(results.keys())
    energies = [results[m]['optimal_energy'] for m in methods]
    reference_energy = results[reference_key]['optimal_energy']
    
    # Plot 1: Energy comparison
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
    bars = ax1.bar(methods, energies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add reference line
    ax1.axhline(
        y=reference_energy,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Reference ({reference_key})',
        alpha=0.7
    )
    
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title('Final Energy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        error = abs(energy - reference_energy)
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{energy:.4f}\n(Δ={error:.4f})',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Plot 2: Convergence overlay
    for method, color in zip(methods, colors):
        if 'energy_history' in results[method]:
            history = results[method]['energy_history']
            iterations = np.arange(1, len(history) + 1)
            ax2.plot(
                iterations,
                history,
                linewidth=2,
                label=method,
                color=color,
                alpha=0.8
            )
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def print_vqe_summary(result: Dict, method_name: str = "VQE"):
    """
    Print a formatted summary of VQE results.
    
    Parameters
    ----------
    result : dict
        VQE result dictionary from run_vqe()
    method_name : str, default="VQE"
        Name of the method for the header
        
    Examples
    --------
    >>> print_vqe_summary(vqe_result, method_name="Exact Statevector")
    """
    print("=" * 70)
    print(f"{method_name} Results")
    print("=" * 70)
    print(f"Optimal Energy:     {result['optimal_energy']:+.8f}")
    print(f"Iterations:         {result['num_iterations']}")
    print(f"Success:            {result['success']}")
    print(f"Optimal Parameters: {result['optimal_params']}")
    if 'optimizer_message' in result:
        print(f"Message:            {result['optimizer_message']}")
    print("=" * 70)
