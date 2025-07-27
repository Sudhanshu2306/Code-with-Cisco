
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_network.nodes import ClassicalNode, QuantumNode
from quantum_network.links import ClassicalLink, QuantumLink, HybridLink
from quantum_network.protocols import HybridRoutingProtocol
from quantum_network.utils import calculate_distance
from network_topology_and_node_simulation.part1_network_topology import HybridNetworkTopology


class HybridRoutingSimulation:
    
    def __init__(self, network: HybridNetworkTopology):
        self.network = network
        self.routing_protocol = HybridRoutingProtocol(
            network.graph, network.nodes, network.links
        )
        self.simulation_results = {}
        
    def design_routing_protocol(self):
        print("Designing hybrid routing protocol...")
        
        self.routing_protocol.build_routing_table()
        
    def simulate_message_routing(self, num_simulations: int = 100):
        print(f"Simulating message routing ({num_simulations} trials)...")
        
        results = {
            "quantum_messages": [],
            "classical_messages": [],
            "adaptive_messages": []
        }
        
        node_list = list(self.network.nodes.keys())
        
        for simulation in range(num_simulations):
            attempts = 0
            while attempts < 10:
                source, destination = random.sample(node_list, 2)
                link_key = (min(source, destination), max(source, destination))
                if link_key not in self.network.links:
                    break
                attempts += 1
            
            if attempts >= 10:
                source, destination = random.sample(node_list, 2)
            
            message_types = [
                ("quantum", {
                    "type": "quantum", 
                    "quantum_state": "|+⟩", 
                    "priority": "high",
                    "requires_fidelity": 0.85
                }),
                ("classical", {
                    "type": "classical", 
                    "data": "Data packet", 
                    "size": random.randint(512, 2048)
                }),
                ("adaptive", {
                    "type": "adaptive", 
                    "data": "Mixed content", 
                    "quantum_component": random.choice([True, False])
                })
            ]
            
            for msg_type, message in message_types:
                routing_result = self.routing_protocol.route_message(
                    source, destination, message, msg_type
                )
                
                if routing_result["success"]:
                    failure_prob = self._calculate_realistic_failure_rate(
                        routing_result, msg_type
                    )
                    if random.random() < failure_prob:
                        routing_result["success"] = False
                        routing_result["error"] = "transmission_failure"
                
                routing_result["source"] = source
                routing_result["destination"] = destination
                routing_result["message_type"] = msg_type
                routing_result["simulation_id"] = simulation
                
                results[f"{msg_type}_messages"].append(routing_result)
        
        self.simulation_results["routing"] = results
        return results
    
    def _calculate_realistic_failure_rate(self, routing_result: Dict, msg_type: str) -> float:
        total_distance = routing_result.get("total_distance", 0)
        num_hops = routing_result.get("hops", 1)
        
        base_rates = {
            "quantum": 0.05,
            "classical": 0.03,
            "adaptive": 0.08
        }
        
        base_rate = base_rates.get(msg_type, 0.10)
        
        if msg_type == "quantum":
            distance_factor = min(total_distance / 1000, 0.25)
            hop_factor = min(num_hops * 0.02, 0.15)
        elif msg_type == "classical":
            distance_factor = min(total_distance / 2500, 0.08)
            hop_factor = min(num_hops * 0.02, 0.06)
        else:
            distance_factor = min(total_distance / 1500, 0.15)
            hop_factor = min(num_hops * 0.04, 0.10)
        
        final_rate = min(base_rate + distance_factor + hop_factor, 0.75)
        return final_rate
        
    def demonstrate_protocol_example(self):
        print("Demonstrating protocol with example: Node A to Node H...")
        
        node_list = list(self.network.nodes.keys())
        if len(node_list) < 2:
            print("Not enough nodes for demonstration")
            return
            
        node_a = node_list[0]
        node_h = node_list[-1]
        
        print(f"Routing from {node_a} to {node_h}")
        
        scenarios = [
            ("High-priority quantum message", {
                "type": "quantum", 
                "quantum_state": "|Ψ⟩ = α|0⟩ + β|1⟩", 
                "priority": "high",
                "requires_entanglement": True
            }),
            ("Classical data packet", {
                "type": "classical", 
                "data": "Important financial transaction", 
                "size": 2048,
                "encryption": "AES-256"
            }),
            ("Hybrid message", {
                "type": "adaptive", 
                "classical_data": "Metadata", 
                "quantum_key": "|key⟩",
                "urgent": True
            })
        ]
        
        example_results = []
        
        for scenario_name, message in scenarios:
            print(f"\nScenario: {scenario_name}")
            print("-" * 40)
            
            quantum_result = self.routing_protocol.route_message(
                node_a, node_h, message, "quantum"
            )
            
            classical_result = self.routing_protocol.route_message(
                node_a, node_h, message, "classical"
            )
            
            adaptive_result = self.routing_protocol.route_message(
                node_a, node_h, message, "adaptive"
            )
            
            results = {
                "scenario": scenario_name,
                "message": message,
                "quantum_routing": quantum_result,
                "classical_routing": classical_result,
                "adaptive_routing": adaptive_result
            }
            
            example_results.append(results)
            
            self._print_routing_result("Quantum-preferred", quantum_result)
            self._print_routing_result("Classical-only", classical_result)
            self._print_routing_result("Adaptive", adaptive_result)
            
            best_approach = self._recommend_best_approach(
                quantum_result, classical_result, adaptive_result, message
            )
            print(f"Recommended approach: {best_approach}")
        
        self.simulation_results["example"] = example_results
        return example_results
        
    def _print_routing_result(self, approach_name: str, result: Dict):
        print(f"\n{approach_name} Routing:")
        if result["success"]:
            print(f"  ✓ Success via {' → '.join(result['path'])}")
            print(f"  ✓ {result['hops']} hops, {result['total_distance']:.1f}km total")
            if "transmission_details" in result:
                details = result["transmission_details"]
                print(f"  ✓ {details['hops_successful']}/{details['hops_attempted']} hops successful")
                print(f"  ✓ Total latency: {details['total_latency']:.4f}s")
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"  ✗ Failed: {error_msg}")
            if "path" in result and result["path"]:
                print(f"    Attempted path: {' → '.join(result['path'])}")
            if "transmission_details" in result:
                details = result["transmission_details"]
                if "failures" in details and details["failures"]:
                    print(f"    Failed at hop {details['failures'][0]['hop']}: {details['failures'][0]['error']}")
            
    def _recommend_best_approach(self, quantum_result: Dict, classical_result: Dict, 
                               adaptive_result: Dict, message: Dict) -> str:
        successful_approaches = []
        
        if quantum_result["success"]:
            successful_approaches.append(("Quantum", quantum_result))
        if classical_result["success"]:
            successful_approaches.append(("Classical", classical_result))
        if adaptive_result["success"]:
            successful_approaches.append(("Adaptive", adaptive_result))
            
        if not successful_approaches:
            return "None (all approaches failed)"
            
        if message.get("type") == "quantum":
            for name, result in successful_approaches:
                if name == "Quantum":
                    return name
                    
        best_approach = min(successful_approaches, 
                          key=lambda x: (x[1]["hops"], x[1]["total_distance"]))
        return best_approach[0]
        
    def analyze_protocol_performance(self):
        print("Analyzing protocol performance...")
        
        routing_results = self.simulation_results["routing"]
        
        success_rates = {}
        path_lengths = {}
        latencies = {}
        
        for msg_type in ["quantum", "classical", "adaptive"]:
            messages = routing_results[f"{msg_type}_messages"]
            successes = sum(1 for msg in messages if msg["success"])
            success_rates[msg_type] = successes / len(messages) if messages else 0
            
            successful_messages = [msg for msg in messages if msg["success"]]
            if successful_messages:
                path_lengths[msg_type] = np.mean([msg["hops"] for msg in successful_messages])
                latencies[msg_type] = np.mean([
                    msg["transmission_details"]["total_latency"] 
                    for msg in successful_messages 
                    if "transmission_details" in msg
                ])
            else:
                path_lengths[msg_type] = 0
                latencies[msg_type] = 0
        
        fallback_analysis = self._analyze_fallback_patterns()
        
        performance_analysis = {
            "success_rates": success_rates,
            "average_path_lengths": path_lengths,
            "average_latencies": latencies,
            "fallback_analysis": fallback_analysis
        }
        
        self.simulation_results["performance"] = performance_analysis
        
        print("\nProtocol Performance Analysis:")
        print("=" * 35)
        for msg_type, success_rate in success_rates.items():
            print(f"{msg_type.capitalize()} messages:")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Average path length: {path_lengths[msg_type]:.1f} hops")
            print(f"  Average latency: {latencies[msg_type]:.4f}s")
            
        return performance_analysis
        
    def _analyze_fallback_patterns(self) -> Dict:
        routing_results = self.simulation_results["routing"]
        
        fallback_stats = {
            "quantum_to_classical": 0,
            "quantum_to_hybrid": 0,
            "direct_quantum_success": 0,
            "direct_classical_success": 0,
            "total_quantum_attempts": 0,
            "total_classical_attempts": 0
        }
        
        quantum_messages = routing_results["quantum_messages"]
        for msg in quantum_messages:
            fallback_stats["total_quantum_attempts"] += 1
            if msg["success"]:
                path_has_classical = False
                for i in range(len(msg["path"]) - 1):
                    node_a, node_b = msg["path"][i], msg["path"][i + 1]
                    link_key = (min(node_a, node_b), max(node_a, node_b))
                    if link_key in self.network.links:
                        link = self.network.links[link_key]
                        if link.type == "Classical":
                            path_has_classical = True
                            break
                            
                if path_has_classical:
                    fallback_stats["quantum_to_classical"] += 1
                else:
                    fallback_stats["direct_quantum_success"] += 1
        
        classical_messages = routing_results["classical_messages"]
        for msg in classical_messages:
            fallback_stats["total_classical_attempts"] += 1
            if msg["success"]:
                fallback_stats["direct_classical_success"] += 1
                
        return fallback_stats
        
    def visualize_protocol_performance(self, save_path: str = "results/hybrid_routing.png"):
        print("Creating protocol visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hybrid Routing Protocol Analysis', fontsize=16, fontweight='bold')
        
        performance = self.simulation_results["performance"]
        
        ax1 = axes[0, 0]
        success_rates = performance["success_rates"]
        message_types = ['Quantum', 'Classical', 'Adaptive']
        rates = [success_rates['quantum'], success_rates['classical'], success_rates['adaptive']]
        colors = ['#E74C3C', '#2ECC71', '#3498DB']
        
        bars = ax1.bar(message_types, rates, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Protocol Success Rates')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax2 = axes[0, 1]
        path_lengths = performance["average_path_lengths"]
        lengths = [path_lengths['quantum'], path_lengths['classical'], path_lengths['adaptive']]
        
        bars = ax2.bar(message_types, lengths, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Average Hops')
        ax2.set_title('Average Path Length')
        ax2.grid(True, alpha=0.3)
        
        for bar, length in zip(bars, lengths):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{length:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax3 = axes[1, 0]
        self._plot_enhanced_network_topology(ax3)
        
        ax4 = axes[1, 1]
        self._plot_performance_metrics(ax4)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Protocol visualization saved to {save_path}")
        
    def _plot_enhanced_network_topology(self, ax):
        pos = {node_id: self.network.positions[node_id] for node_id in self.network.graph.nodes()}
        
        quantum_edges = []
        classical_edges = []
        hybrid_edges = []
        
        for edge in self.network.graph.edges():
            link_key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if link_key in self.network.links:
                link = self.network.links[link_key]
                if hasattr(link, 'type'):
                    if link.type == "Quantum":
                        quantum_edges.append(edge)
                    elif link.type == "Classical":
                        classical_edges.append(edge)
                    else:
                        hybrid_edges.append(edge)
        
        if quantum_edges:
            nx.draw_networkx_edges(self.network.graph, pos, edgelist=quantum_edges,
                                 edge_color='#3498DB', width=2, alpha=0.7, ax=ax)
        if classical_edges:
            nx.draw_networkx_edges(self.network.graph, pos, edgelist=classical_edges,
                                 edge_color='#E74C3C', width=2, alpha=0.7, ax=ax)
        if hybrid_edges:
            nx.draw_networkx_edges(self.network.graph, pos, edgelist=hybrid_edges,
                                 edge_color='#F39C12', width=2, alpha=0.7, ax=ax)
        
        node_colors = {"Classical": "#E74C3C", "Quantum": "#3498DB", "Hybrid": "#F39C12"}
        for node_id, node_data in self.network.graph.nodes(data=True):
            node_type = node_data.get('type', 'Classical')
            color = node_colors.get(node_type, 'gray')
            x, y = pos[node_id]
            ax.scatter(x, y, c=color, s=200, alpha=0.9, edgecolors='black', linewidth=2)
            ax.text(x, y, node_id.split('_')[1], ha='center', va='center', 
                   fontweight='bold', fontsize=8, color='white')
            
        ax.set_title('Network Topology')
        ax.set_xlabel('X Position (km)')
        ax.set_ylabel('Y Position (km)')
        
        legend_elements = [
            plt.Line2D([0], [0], color='#3498DB', lw=3, label='Quantum Links'),
            plt.Line2D([0], [0], color='#E74C3C', lw=3, label='Classical Links'),
            plt.Line2D([0], [0], color='#F39C12', lw=3, label='Hybrid Links')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
    def _plot_performance_metrics(self, ax):
        performance = self.simulation_results["performance"]
        
        metrics = ['Success Rate', 'Avg Path Length', 'Avg Latency (ms)']
        quantum_vals = [
            performance["success_rates"]["quantum"],
            performance["average_path_lengths"]["quantum"] / 5,
            performance["average_latencies"]["quantum"] * 1000
        ]
        classical_vals = [
            performance["success_rates"]["classical"],
            performance["average_path_lengths"]["classical"] / 5,
            performance["average_latencies"]["classical"] * 1000
        ]
        adaptive_vals = [
            performance["success_rates"]["adaptive"],
            performance["average_path_lengths"]["adaptive"] / 5,
            performance["average_latencies"]["adaptive"] * 1000
        ]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        bars1 = ax.bar(x - width, quantum_vals, width, label='Quantum', 
                      color='#3498DB', alpha=0.8)
        bars2 = ax.bar(x, classical_vals, width, label='Classical', 
                      color='#E74C3C', alpha=0.8)
        bars3 = ax.bar(x + width, adaptive_vals, width, label='Adaptive', 
                      color='#F39C12', alpha=0.8)
        
        ax.set_ylabel('Normalized Values')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)


def main():
    print("Quantum-Classical Hybrid Network Simulation")
    print("=" * 50)
    print("Part 3: Protocol Design for Hybrid Routing")
    print()
    
    print("Creating network topology...")
    network = HybridNetworkTopology(num_nodes=12)
    network.create_network()
    
    node_list = list(network.nodes.keys())
    links_to_remove = []
    
    for link_key, link in network.links.items():
        if link.distance > 400:
            links_to_remove.append(link_key)
    
    for link_key in links_to_remove:
        if link_key in network.links:
            node_a, node_b = link_key
            del network.links[link_key]
            if network.graph.has_edge(node_a, node_b):
                network.graph.remove_edge(node_a, node_b)
    
    for i in range(len(node_list)):
        for j in range(i + 1, min(i + 4, len(node_list))):
            node_a, node_b = node_list[i], node_list[j]
            link_key = (min(node_a, node_b), max(node_a, node_b))
            
            if link_key not in network.links:
                distance = calculate_distance(network.positions[node_a], network.positions[node_b])
                if distance < 300:
                    link = network._create_appropriate_link(node_a, node_b, distance)
                    network.links[link_key] = link
                    network.graph.add_edge(node_a, node_b,
                                         link_type=link.type,
                                         distance=distance,
                                         link_obj=link)
                    network.nodes[node_a].add_connection(node_b)
                    network.nodes[node_b].add_connection(node_a)
    
    print(f"Enhanced network: {len(network.nodes)} nodes, {len(network.links)} links")
    
    routing_sim = HybridRoutingSimulation(network)
    
    protocol_description = routing_sim.design_routing_protocol()
    
    routing_sim.simulate_message_routing(50)
    
    routing_sim.demonstrate_protocol_example()
    
    routing_sim.analyze_protocol_performance()
    
    routing_sim.visualize_protocol_performance()
    
    with open("results/hybrid_routing_protocol.txt", "w") as f:
        f.write("\n\nPerformance Analysis:\n")
        f.write("==================\n")
        performance = routing_sim.simulation_results["performance"]
        for key, value in performance.items():
            f.write(f"{key}: {value}\n")
    
    print("\nPart 3 completed successfully!")
    print("Hybrid routing protocol has been designed, implemented, and analyzed.")
    
    return routing_sim

if __name__ == "__main__":
    routing_sim = main()
