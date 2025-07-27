"""
Utility functions for quantum network simulation.

This module provides helper functions for calculations and common operations
used throughout the quantum network simulation.
"""

import numpy as np
import math
from typing import Tuple, List, Dict, Any

def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two positions.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        Distance between the positions
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def quantum_decoherence_probability(time: float, decoherence_rate: float) -> float:
    """
    Calculate probability of quantum decoherence over time.
    
    Args:
        time: Time duration
        decoherence_rate: Rate of decoherence
        
    Returns:
        Probability of decoherence (0 to 1)
    """
    return 1 - np.exp(-decoherence_rate * time)


def quantum_fidelity_decay(initial_fidelity: float, distance: float, 
                          decay_constant: float = 0.01) -> float:
    """
    Calculate quantum fidelity decay over distance.
    
    Args:
        initial_fidelity: Starting fidelity
        distance: Transmission distance
        decay_constant: Decay rate constant
        
    Returns:
        Final fidelity after decay
    """
    return initial_fidelity * np.exp(-decay_constant * distance)


def calculate_path_length(path: List[str], positions: Dict[str, Tuple[float, float]]) -> float:
    """
    Calculate total path length given a sequence of nodes.
    
    Args:
        path: List of node IDs representing the path
        positions: Dictionary mapping node IDs to positions
        
    Returns:
        Total path length
    """
    if len(path) < 2:
        return 0.0
        
    total_length = 0.0
    for i in range(len(path) - 1):
        pos1 = positions[path[i]]
        pos2 = positions[path[i + 1]]
        total_length += calculate_distance(pos1, pos2)
        
    return total_length


def quantum_entanglement_success_probability(distance: float, 
                                           base_probability: float = 0.9,
                                           decay_length: float = 100.0) -> float:
    """
    Calculate probability of successful entanglement establishment.
    
    Args:
        distance: Distance between nodes
        base_probability: Base success probability at distance 0
        decay_length: Characteristic decay length
        
    Returns:
        Success probability (0 to 1)
    """
    return base_probability * np.exp(-distance / decay_length)


def classical_packet_loss_probability(distance: float, base_loss: float = 0.001,
                                     distance_factor: float = 0.0001) -> float:
    """
    Calculate classical packet loss probability.
    
    Args:
        distance: Link distance
        base_loss: Base loss probability
        distance_factor: Additional loss per unit distance
        
    Returns:
        Packet loss probability (0 to 1)
    """
    return min(1.0, base_loss + distance_factor * distance)


def generate_random_network_positions(num_nodes: int, width: float = 1000.0, 
                                    height: float = 1000.0) -> Dict[str, Tuple[float, float]]:
    """
    Generate random positions for network nodes.
    
    Args:
        num_nodes: Number of nodes
        width: Width of the area
        height: Height of the area
        
    Returns:
        Dictionary mapping node IDs to positions
    """
    positions = {}
    for i in range(num_nodes):
        node_id = f"Node_{i}"
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        positions[node_id] = (x, y)
        
    return positions


def calculate_network_connectivity(adjacency_matrix: np.ndarray) -> float:
    """
    Calculate network connectivity metric.
    
    Args:
        adjacency_matrix: Network adjacency matrix
        
    Returns:
        Connectivity metric (0 to 1)
    """
    n = len(adjacency_matrix)
    if n <= 1:
        return 1.0
        
    total_possible_edges = n * (n - 1) / 2
    actual_edges = np.sum(adjacency_matrix) / 2  # Undirected graph
    
    return actual_edges / total_possible_edges


def simulate_noise(signal_strength: float, noise_level: float = 0.1) -> float:
    """
    Simulate noise in signal transmission.
    
    Args:
        signal_strength: Original signal strength
        noise_level: Standard deviation of noise
        
    Returns:
        Noisy signal strength
    """
    noise = np.random.normal(0, noise_level)
    return max(0, signal_strength + noise)


def quantum_error_correction_threshold(error_rate: float, 
                                     threshold: float = 0.11) -> bool:
    """
    Check if quantum error correction is possible.
    
    Args:
        error_rate: Current error rate
        threshold: Error correction threshold
        
    Returns:
        True if error correction is viable
    """
    return error_rate < threshold


def calculate_quantum_capacity(distance: float, base_rate: float = 1000.0,
                             attenuation_length: float = 22.0) -> float:
    """
    Calculate quantum channel capacity based on distance.
    
    Args:
        distance: Channel distance in km
        base_rate: Base transmission rate in qubits/second
        attenuation_length: Characteristic attenuation length in km
        
    Returns:
        Channel capacity in qubits/second
    """
    return base_rate * np.exp(-distance / attenuation_length)


def bell_state_fidelity(noise_level: float = 0.05) -> float:
    """
    Calculate Bell state fidelity with noise.
    
    Args:
        noise_level: Noise level affecting the Bell state
        
    Returns:
        Fidelity of the Bell state (0 to 1)
    """
    ideal_fidelity = 1.0
    noise_factor = np.random.exponential(noise_level)
    return max(0, ideal_fidelity - noise_factor)


def entanglement_swapping_success_rate(intermediate_fidelity: float,
                                      measurement_efficiency: float = 0.9) -> float:
    """
    Calculate success rate of entanglement swapping.
    
    Args:
        intermediate_fidelity: Fidelity of intermediate entangled pairs
        measurement_efficiency: Efficiency of Bell state measurement
        
    Returns:
        Success rate of entanglement swapping
    """
    return intermediate_fidelity * measurement_efficiency


def format_network_statistics(stats: Dict[str, Any]) -> str:
    """
    Format network statistics for display.
    
    Args:
        stats: Dictionary of network statistics
        
    Returns:
        Formatted string representation
    """
    formatted = "Network Statistics:\n"
    formatted += "==================\n"
    
    for key, value in stats.items():
        if isinstance(value, float):
            formatted += f"{key}: {value:.4f}\n"
        else:
            formatted += f"{key}: {value}\n"
            
    return formatted
