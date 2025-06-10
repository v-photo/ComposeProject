"""
Physics-Informed Neural Networks (PINN) for radiation dose simulation
Package initialization
"""

from .tools import (
    SimulationConfig,
    GPUMonteCarloSimulator,
    RadiationDataProcessor,
    DataLoader, 
    PINNTrainer,
    ResultAnalyzer,
    Visualizer,
    setup_deepxde_backend
)

__version__ = "1.0.0"
__author__ = "PINN Team"

__all__ = [
    'SimulationConfig',
    'GPUMonteCarloSimulator',
    'RadiationDataProcessor', 
    'DataLoader',
    'PINNTrainer',
    'ResultAnalyzer',
    'Visualizer',
    'setup_deepxde_backend'
] 