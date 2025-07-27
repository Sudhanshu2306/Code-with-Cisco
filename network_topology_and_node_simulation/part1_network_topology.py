import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from typing import Dict, List, Tuple
import os

from quantum_network.nodes import ClassicalNode, QuantumNode
from quantum_network.links import ClassicalLink, QuantumLink, HybridLink
from quantum_network.utils import calculate_distance, generate_random_network_positions


class HybridNetworkTopology:
    """Creates and manages a quantum-classical hybrid network topology."""
    
    def __init__(self, num_nodes: int = 15):
        self.num_nodes = num_nodes
        self.graph = nx.Graph()
        self.nodes: Dict[str, object] = {}
        self.links: Dict[Tuple[str, str], object] = {}
        self.positions: Dict[str, Tuple[float, float]] = {}
        
    def create_network(self):
        """Create the hybrid network with different node types."""
        print(f"Creating hybrid network with {self.num_nodes} nodes...")
        
        # Generate node positions
        self.positions = generate_random_network_positions(self.num_nodes, 1000, 1000)
        
        # Create nodes with different types
        node_types = self._determine_node_types()
        
        for i, (node_id, node_type) in enumerate(node_types.items()):
            position = self.positions[node_id]
            
            if node_type == "classical":
                node = ClassicalNode(node_id, position)
            elif node_type == "quantum":
                node = QuantumNode(node_id, position)
            else:  
                node = QuantumNode(node_id, position)
                
            self.nodes[node_id] = node
            self.graph.add_node(node_id, 
                              type=node_type,
                              position=position,
                              node_obj=node)
        
        # Create links between nodes
        self._create_links()
        
        print(f"Network created with {len(self.nodes)} nodes and {len(self.links)} links")
        # self._print_network_summary()
        
    def _determine_node_types(self) -> Dict[str, str]:
        """Determine the type of each node."""
        node_types = {}
        
        # Ensure we have at least 2 of each type
        guaranteed_types = ["classical", "classical", "quantum", "quantum", "hybrid", "hybrid"]
        
        # Assign remaining nodes randomly
        remaining_count = self.num_nodes - len(guaranteed_types)
        type_choices = ["classical", "quantum", "hybrid"]
        remaining_types = [random.choice(type_choices) for _ in range(remaining_count)]
        
        all_types = guaranteed_types + remaining_types
        random.shuffle(all_types)
        
        for i in range(self.num_nodes):
            node_id = f"Node_{i}"
            node_types[node_id] = all_types[i]
            
        return node_types
        
    def _create_links(self):
        """Create links between nodes based on proximity and capabilities."""
        # Calculate distances between all node pairs
        node_pairs = []
        for i, node_a in enumerate(self.nodes.keys()):
            for j, node_b in enumerate(list(self.nodes.keys())[i+1:], i+1):
                distance = calculate_distance(self.positions[node_a], self.positions[node_b])
                node_pairs.append((node_a, node_b, distance))
        
        # Sort by distance
        node_pairs.sort(key=lambda x: x[2])
        
        # Create links ensuring network connectivity
        connected_components = [{node} for node in self.nodes.keys()]
        
        # First, ensure connectivity by creating a minimum spanning tree
        for node_a, node_b, distance in node_pairs:
            # Find components containing node_a and node_b
            comp_a = next((i for i, comp in enumerate(connected_components) if node_a in comp), None)
            comp_b = next((i for i, comp in enumerate(connected_components) if node_b in comp), None)
            
            if comp_a != comp_b:  # Nodes in different components
                # Merge components
                connected_components[comp_a].update(connected_components[comp_b])
                del connected_components[comp_b]
                
                # Create appropriate link
                link = self._create_appropriate_link(node_a, node_b, distance)
                self.links[(node_a, node_b)] = link
                self.graph.add_edge(node_a, node_b, 
                                  link_type=link.type,
                                  distance=distance,
                                  link_obj=link)
                
                # Update node connections
                self.nodes[node_a].add_connection(node_b)
                self.nodes[node_b].add_connection(node_a)
                
                if len(connected_components) == 1:  # All connected
                    break
        
        # Add additional links for redundancy (up to 2x minimum connections)
        max_additional_links = len(self.nodes) - 1  # Allow up to 2x MST edges
        additional_links = 0
        
        for node_a, node_b, distance in node_pairs:
            if additional_links >= max_additional_links:
                break
                
            # Don't create link if already exists
            if (node_a, node_b) in self.links or (node_b, node_a) in self.links:
                continue
                
            # Don't create very long links (more than 500 units)
            if distance > 500:
                continue
                
            # Create link with some probability based on distance
            connection_prob = max(0, 1 - distance / 300)
            if random.random() < connection_prob:
                link = self._create_appropriate_link(node_a, node_b, distance)
                self.links[(node_a, node_b)] = link
                self.graph.add_edge(node_a, node_b,
                                  link_type=link.type,
                                  distance=distance,
                                  link_obj=link)
                
                self.nodes[node_a].add_connection(node_b)
                self.nodes[node_b].add_connection(node_a)
                additional_links += 1
                
    def _create_appropriate_link(self, node_a: str, node_b: str, distance: float):
        """Create appropriate link type based on node capabilities."""
        node_a_obj = self.nodes[node_a]
        node_b_obj = self.nodes[node_b]
        
        node_a_type = node_a_obj.type
        node_b_type = node_b_obj.type
        
        # Determine link type based on node capabilities
        if node_a_type == "Classical" and node_b_type == "Classical":
            return ClassicalLink(node_a, node_b, distance)
        elif (node_a_type == "Quantum" and node_b_type == "Quantum"):
            return QuantumLink(node_a, node_b, distance)
        elif ((node_a_type == "Quantum" or node_a_type == "Hybrid") and 
              (node_b_type == "Quantum" or node_b_type == "Hybrid")):
            # At least one quantum-capable node - use quantum link
            return QuantumLink(node_a, node_b, distance)
        else:
            # Mixed capabilities - use hybrid link
            return HybridLink(node_a, node_b, distance)
            
    def _calculate_network_diameter(self) -> float:
        """Calculate the diameter of the network."""
        try:
            # Use NetworkX to find shortest paths
            path_lengths = dict(nx.all_pairs_shortest_path_length(self.graph))
            max_distance = 0
            
            for source in path_lengths:
                for target in path_lengths[source]:
                    if path_lengths[source][target] > max_distance:
                        max_distance = path_lengths[source][target]
                        
            return max_distance
        except:
            return 0.0
            
    def visualize_network(self, save_path: str = "results/network_topology.png"):
        """Create visualization of the network topology."""
        print("Creating network visualization...")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Define colors for different node types
        node_colors = {
            "Classical": "#FF6B6B",  # Red
            "Quantum": "#4ECDC4",    # Teal
            "Hybrid": "#45B7D1",     # Blue
            "classical": "#FF6B6B",  # Red
            "quantum": "#4ECDC4",    # Teal
            "hybrid": "#45B7D1"      # Blue
        }
        
        # Define colors for different link types
        link_colors = {
            "Classical": "#FF6B6B",  # Red
            "Quantum": "#4ECDC4",    # Teal
            "Hybrid": "#45B7D1",     # Blue
            "classical": "#FF6B6B",  # Red
            "quantum": "#4ECDC4",    # Teal
            "hybrid": "#45B7D1"      # Blue
        }
        
        # Plot 1: Network topology with node types
        pos = {node_id: self.positions[node_id] for node_id in self.graph.nodes()}
        
        # Draw links first (so they appear behind nodes)
        for edge in self.graph.edges(data=True):
            node_a, node_b, data = edge
            link_type = data['link_type']
            color = link_colors[link_type]
            
            x_coords = [pos[node_a][0], pos[node_b][0]]
            y_coords = [pos[node_a][1], pos[node_b][1]]
            
            if link_type == "Quantum":
                ax1.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8, linestyle='-')
            elif link_type == "Hybrid":
                ax1.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8, linestyle='--')
            else:  # Classical
                ax1.plot(x_coords, y_coords, color=color, linewidth=1.5, alpha=0.6, linestyle=':')
        
        # Draw nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data['type']
            color = node_colors[node_type]
            x, y = pos[node_id]
            
            if node_type == "Quantum":
                marker = 's'  # Square
                size = 200
            elif node_type == "Hybrid":
                marker = 'D'  # Diamond
                size = 180
            else:  # Classical
                marker = 'o'  # Circle
                size = 150
                
            ax1.scatter(x, y, c=color, marker=marker, s=size, 
                       edgecolors='black', linewidths=1, alpha=0.8, zorder=3)
            ax1.annotate(node_id.replace("Node_", ""), (x, y), 
                        xytext=(0, 0), textcoords='offset points',
                        ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax1.set_title("Quantum-Classical Hybrid Network Topology", fontsize=14, fontweight='bold')
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.grid(True, alpha=0.3)
        
        # Create legend for nodes
        legend_elements_nodes = [
            plt.scatter([], [], c=node_colors["Classical"], marker='o', s=150, 
                       edgecolors='black', label='Classical Node'),
            plt.scatter([], [], c=node_colors["Quantum"], marker='s', s=200, 
                       edgecolors='black', label='Quantum Node'),
            
        ]
        
        # Create legend for links
        legend_elements_links = [
            plt.Line2D([0], [0], color=link_colors["Classical"], linewidth=1.5, 
                      linestyle=':', label='Classical Link'),
            plt.Line2D([0], [0], color=link_colors["Quantum"], linewidth=2, 
                      linestyle='-', label='Quantum Link'),
            plt.Line2D([0], [0], color=link_colors["Hybrid"], linewidth=2, 
                      linestyle='--', label='Hybrid Link')
        ]
        
        # Add legends
        legend1 = ax1.legend(handles=legend_elements_nodes, title="Node Types", 
                           loc='upper left', bbox_to_anchor=(0, 1))
        legend2 = ax1.legend(handles=legend_elements_links, title="Link Types", 
                           loc='upper left', bbox_to_anchor=(0, 0.7))
        ax1.add_artist(legend1)
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Network visualization saved to {save_path}")
        
        
    def get_network_stats(self) -> Dict:
        """Get comprehensive network statistics."""
        classical_nodes = sum(1 for node in self.nodes.values() if node.type == "Classical")
        quantum_nodes = sum(1 for node in self.nodes.values() if node.type == "Quantum")
        hybrid_nodes = sum(1 for node in self.nodes.values() if node.type == "Hybrid")
        
        classical_links = sum(1 for link in self.links.values() if link.type == "Classical")
        quantum_links = sum(1 for link in self.links.values() if link.type == "Quantum")
        hybrid_links = sum(1 for link in self.links.values() if link.type == "Hybrid")
        
        degrees = [len(node.connected_nodes) for node in self.nodes.values()]
        distances = [link.distance for link in self.links.values()]
        
        return {
            "total_nodes": len(self.nodes),
            "classical_nodes": classical_nodes,
            "quantum_nodes": quantum_nodes,
            "hybrid_nodes": hybrid_nodes,
            "total_links": len(self.links),
            "classical_links": classical_links,
            "quantum_links": quantum_links,
            "hybrid_links": hybrid_links,
            "avg_degree": np.mean(degrees),
            "max_degree": np.max(degrees),
            "min_degree": np.min(degrees),
            "avg_distance": np.mean(distances),
            "max_distance": np.max(distances),
            "min_distance": np.min(distances),
            "is_connected": nx.is_connected(self.graph),
            "diameter": self._calculate_network_diameter()
        }


def main():
    """Main function to create and visualize the hybrid network."""
    print("Quantum-Classical Hybrid Network Simulation")
    print("=" * 50)
    print("Part 1: Network Topology and Node Simulation")
    print()
    
    # Create network
    network = HybridNetworkTopology(num_nodes=15)
    network.create_network()
    
    # Visualize network
    network.visualize_network()
    
    return network


if __name__ == "__main__":
    network = main()
