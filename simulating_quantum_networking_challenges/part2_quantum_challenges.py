import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional
import os

from quantum_network.nodes import ClassicalNode, QuantumNode
from quantum_network.links import ClassicalLink, QuantumLink, HybridLink
from quantum_network.utils import (quantum_decoherence_probability, 
                                  quantum_fidelity_decay, 
                                  calculate_quantum_capacity)


class QuantumNetworkChallenges:
    
    def __init__(self):
        self.simulation_results = {}
        self.distances = np.linspace(10, 500, 50)  # km
        self.num_trials = 1000
        
    def simulate_decoherence(self) -> Dict:
        
        results = {
            "distances": [],
            "success_rates": [],
            "fidelities": [],
            "transmission_times": []
        }
        
        for distance in self.distances:
            successful_transmissions = 0
            total_fidelity = 0
            total_time = 0
            
            for trial in range(self.num_trials):
                transmission_time = distance / 200000 
                
                decoherence_rate = 0.1 
                decoherence_prob = quantum_decoherence_probability(transmission_time, decoherence_rate)
                
                if random.random() > decoherence_prob:
                    successful_transmissions += 1
                    initial_fidelity = 0.99
                    final_fidelity = quantum_fidelity_decay(initial_fidelity, distance, 0.005)
                    total_fidelity += final_fidelity
                
                total_time += transmission_time
            
            success_rate = successful_transmissions / self.num_trials
            avg_fidelity = total_fidelity / max(1, successful_transmissions)
            avg_time = total_time / self.num_trials
            
            results["distances"].append(distance)
            results["success_rates"].append(success_rate)
            results["fidelities"].append(avg_fidelity)
            results["transmission_times"].append(avg_time)
        
        self.simulation_results["decoherence"] = results
        return results
        
    def simulate_no_cloning(self) -> Dict:
        print("Simulating no-cloning theorem effects...")
        
        results = {
            "attempt_types": [],
            "success_rates": [],
            "error_types": []
        }
        
        scenarios = [
            ("direct_transmission", "Direct quantum transmission"),
            ("attempted_amplification", "Attempted signal amplification"),
            ("attempted_copying", "Attempted quantum state copying"),
            ("measurement_copy", "Copy via measurement")
        ]
        
        for scenario_type, description in scenarios:
            successful_transmissions = 0
            error_counts = {"success": 0, "amplification_failed": 0, 
                          "copy_failed": 0, "measurement_destroyed": 0}
            
            for trial in range(self.num_trials):
                if scenario_type == "direct_transmission":
                    # Direct transmission - should work with natural loss
                    if random.random() > 0.1:  # 90% success for direct transmission
                        successful_transmissions += 1
                        error_counts["success"] += 1
                    else:
                        error_counts["measurement_destroyed"] += 1
                        
                elif scenario_type == "attempted_amplification":
                    # Amplification always fails due to no-cloning
                    error_counts["amplification_failed"] += 1
                    
                elif scenario_type == "attempted_copying":
                    # Copying always fails due to no-cloning
                    error_counts["copy_failed"] += 1
                    
                elif scenario_type == "measurement_copy":
                    # Measurement destroys the original state
                    error_counts["measurement_destroyed"] += 1
            
            success_rate = successful_transmissions / self.num_trials
            results["attempt_types"].append(description)
            results["success_rates"].append(success_rate)
            results["error_types"].append(error_counts)
        
        self.simulation_results["no_cloning"] = results
        return results
        
    def simulate_entanglement_distribution(self) -> Dict:
        """Simulate entanglement distribution and swapping."""
        print("Simulating entanglement distribution...")
        
        results = {
            "network_sizes": [],
            "direct_success_rates": [],
            "swapped_success_rates": [],
            "end_to_end_fidelities": []
        }
        
        network_sizes = range(3, 21)  # Networks from 3 to 20 nodes
        
        for network_size in network_sizes:
            direct_successes = 0
            swapped_successes = 0
            total_fidelity = 0
            
            for trial in range(self.num_trials):
                total_distance = (network_size - 1) * 100  # Assume 100km per hop
                
                direct_prob = np.exp(-total_distance / 50)
                if random.random() < direct_prob:
                    direct_successes += 1
                
                hop_success_prob = 0.8  # 80% success per hop
                swap_success = True
                current_fidelity = 0.95  # Initial fidelity
                
                for hop in range(network_size - 1):
                    if random.random() > hop_success_prob:
                        swap_success = False
                        break
                    # Fidelity degrades with each swap
                    current_fidelity *= 0.95
                
                if swap_success:
                    swapped_successes += 1
                    total_fidelity += current_fidelity
            
            direct_success_rate = direct_successes / self.num_trials
            swapped_success_rate = swapped_successes / self.num_trials
            avg_fidelity = total_fidelity / max(1, swapped_successes)
            
            results["network_sizes"].append(network_size)
            results["direct_success_rates"].append(direct_success_rate)
            results["swapped_success_rates"].append(swapped_success_rate)
            results["end_to_end_fidelities"].append(avg_fidelity)
        
        self.simulation_results["entanglement"] = results
        return results
        
    def simulate_classical_comparison(self) -> Dict:
        print("Simulating classical networking for comparison...")
        
        results = {
            "distances": [],
            "packet_loss_rates": [],
            "latencies": [],
            "throughputs": []
        }
        
        for distance in self.distances:
            packet_losses = 0
            total_latency = 0
            successful_packets = 0
            
            for trial in range(self.num_trials):
                # Classical packet loss increases with distance
                base_loss_rate = 0.001
                distance_loss_factor = distance * 0.00001
                loss_rate = base_loss_rate + distance_loss_factor
                
                if random.random() > loss_rate:
                    successful_packets += 1
                    
                    # Calculate latency (speed of light + processing delays)
                    propagation_delay = distance / 200000  # seconds
                    processing_delay = np.random.exponential(0.001)  # Random processing delay
                    total_latency += propagation_delay + processing_delay
                else:
                    packet_losses += 1
            
            packet_loss_rate = packet_losses / self.num_trials
            avg_latency = total_latency / max(1, successful_packets)
            throughput = successful_packets / (avg_latency * self.num_trials) if avg_latency > 0 else 0
            
            results["distances"].append(distance)
            results["packet_loss_rates"].append(packet_loss_rate)
            results["latencies"].append(avg_latency)
            results["throughputs"].append(throughput)
        
        self.simulation_results["classical"] = results
        return results
        
    # def analyze_scaling_effects(self) -> Dict:
    #     """Analyze how quantum effects scale with network size and distance."""
    #     print("Analyzing scaling effects...")
        
    #     results = {
    #         "quantum_capacity": [],
    #         "classical_capacity": [],
    #         "quantum_error_rates": [],
    #         "classical_error_rates": [],
    #         "distances": list(self.distances)
    #     }
        
    #     for distance in self.distances:
    #         # Quantum capacity decreases exponentially
    #         quantum_cap = calculate_quantum_capacity(distance, 1000, 22)
            
    #         # Classical capacity decreases more slowly
    #         classical_cap = 1000 * (1 - distance * 0.0001)  # Linear decrease
            
    #         # Quantum error rates increase with distance
    #         quantum_error = 1 - np.exp(-distance / 100)
            
    #         # Classical error rates increase more slowly
    #         classical_error = min(0.1, distance * 0.0001)
            
    #         results["quantum_capacity"].append(quantum_cap)
    #         results["classical_capacity"].append(max(0, classical_cap))
    #         results["quantum_error_rates"].append(quantum_error)
    #         results["classical_error_rates"].append(classical_error)
        
    #     self.simulation_results["scaling"] = results
    #     return results
        
    def visualize_results(self, save_path: str = "results/quantum_challenges.png"):
        """Create comprehensive visualization of simulation results."""
        print("Creating visualization...")
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quantum Networking Challenges Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Decoherence effects
        ax1 = axes[0, 0]
        decoherence_data = self.simulation_results["decoherence"]
        ax1.plot(decoherence_data["distances"], decoherence_data["success_rates"], 
                'b-', linewidth=2, label='Success Rate')
        ax1.plot(decoherence_data["distances"], decoherence_data["fidelities"], 
                'r--', linewidth=2, label='Average Fidelity')
        ax1.set_xlabel('Distance (km)')
        ax1.set_ylabel('Rate / Fidelity')
        ax1.set_title('Quantum Decoherence Effects')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: No-cloning violations
        ax2 = axes[0, 1]
        no_cloning_data = self.simulation_results["no_cloning"]
        bars = ax2.bar(range(len(no_cloning_data["attempt_types"])), 
                      no_cloning_data["success_rates"], 
                      color=['green', 'red', 'red', 'orange'])
        ax2.set_xlabel('Transmission Type')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('No-Cloning Theorem Effects')
        ax2.set_xticks(range(len(no_cloning_data["attempt_types"])))
        ax2.set_xticklabels([t.split()[0] for t in no_cloning_data["attempt_types"]], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Entanglement distribution
        ax3 = axes[0, 2]
        entanglement_data = self.simulation_results["entanglement"]
        ax3.plot(entanglement_data["network_sizes"], entanglement_data["direct_success_rates"], 
                'r-o', linewidth=2, label='Direct Entanglement')
        ax3.plot(entanglement_data["network_sizes"], entanglement_data["swapped_success_rates"], 
                'b-s', linewidth=2, label='Entanglement Swapping')
        ax3.set_xlabel('Network Size (nodes)')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Entanglement Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Quantum vs Classical comparison
        ax4 = axes[1, 0]
        classical_data = self.simulation_results["classical"]
        ax4.plot(decoherence_data["distances"], 
                [1 - rate for rate in decoherence_data["success_rates"]], 
                'r-', linewidth=2, label='Quantum Loss Rate')
        ax4.plot(classical_data["distances"], classical_data["packet_loss_rates"], 
                'b--', linewidth=2, label='Classical Loss Rate')
        ax4.set_xlabel('Distance (km)')
        ax4.set_ylabel('Loss Rate')
        ax4.set_title('Quantum vs Classical Loss Rates')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to {save_path}")
        
    def _get_value_at_distance(self, category: str, metric: str, distance: float) -> float:
        data = self.simulation_results[category]
        distances = data["distances"]
        values = data[metric]
        
        # Find closest distance
        idx = min(range(len(distances)), key=lambda i: abs(distances[i] - distance))
        return values[idx]
        
    def _find_critical_distance(self, category: str, metric: str, threshold: float) -> float:
        """Find the distance at which a metric crosses a threshold."""
        if category not in self.simulation_results:
            return 0.0  # Return default if category doesn't exist
        data = self.simulation_results[category]
        distances = data["distances"]
        values = data[metric]
        
        for i, value in enumerate(values):
            if value <= threshold:
                return distances[i]
        return distances[-1]  # Return max distance if threshold not reached


def main():
    """Main function to run quantum networking challenges simulation."""
    print("Quantum-Classical Hybrid Network Simulation")
    print("=" * 50)
    print("Part 2: Simulating Quantum Networking Challenges")
    print()
    
    # Create and run simulations
    simulator = QuantumNetworkChallenges()
    
    # Run all simulations
    simulator.simulate_decoherence()
    simulator.simulate_no_cloning()
    simulator.simulate_entanglement_distribution()
    simulator.simulate_classical_comparison()
    # simulator.analyze_scaling_effects()
    
    # Visualize results
    simulator.visualize_results()
    
    # Generate and print report
    report = simulator.generate_report()
    print(report)
    
    # Save report to file
    with open("results/quantum_challenges_report.txt", "w") as f:
        f.write(report)
    
    print("\nPart 2 completed successfully!")
    print("Quantum networking challenges have been analyzed and visualized.")
    
    return simulator


if __name__ == "__main__":
    simulator = main()
