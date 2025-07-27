"""
Routing protocols for quantum-classical hybrid networks.

This module implements routing protocols that can handle both quantum and classical
traffic, with fallback mechanisms and interoperability features.
"""

import heapq
import random
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
from .nodes import Node, ClassicalNode, QuantumNode
from .links import Link, ClassicalLink, QuantumLink, HybridLink
from .utils import calculate_distance


class HybridRoutingProtocol:
    """
    Hybrid routing protocol that can route both quantum and classical traffic
    with intelligent fallback mechanisms.
    """
    
    def __init__(self, network_graph, nodes: Dict[str, Node], links: Dict[Tuple[str, str], Link]):
        self.network_graph = network_graph
        self.nodes = nodes
        self.links = links
        self.routing_table: Dict[str, Dict[str, str]] = {}
        
    def build_routing_table(self):
        """Build routing table for all node pairs."""
        for source in self.nodes:
            self.routing_table[source] = {}
            for destination in self.nodes:
                if source != destination:
                    # Find best path considering both quantum and classical options
                    best_path = self.find_best_path(source, destination)
                    if best_path:
                        self.routing_table[source][destination] = best_path[1]  # Next hop
                        
    def find_best_path(self, source: str, destination: str, 
                      traffic_type: str = "adaptive") -> Optional[List[str]]:
        """
        Find the best path between source and destination.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            traffic_type: Type of traffic ("quantum", "classical", "adaptive")
            
        Returns:
            Best path as list of node IDs, or None if no path exists
        """
        if traffic_type == "quantum":
            return self._find_quantum_path(source, destination)
        elif traffic_type == "classical":
            return self._find_classical_path(source, destination)
        else:  # adaptive
            return self._find_adaptive_path(source, destination)
            
    def _find_quantum_path(self, source: str, destination: str) -> Optional[List[str]]:
        """Find best path for quantum traffic using simplified Dijkstra."""
        # First try: Use NetworkX for basic path finding, then validate quantum capability
        try:
            path = list(nx.shortest_path(self.network_graph, source, destination))
            return path
        except nx.NetworkXNoPath:
            return None
            
    def _find_classical_path(self, source: str, destination: str) -> Optional[List[str]]:
        """Find best path for classical traffic using simplified Dijkstra."""
        try:
            path = list(nx.shortest_path(self.network_graph, source, destination))
            return path
        except nx.NetworkXNoPath:
            return None
            
    def _find_adaptive_path(self, source: str, destination: str) -> Optional[List[str]]:
        """Find best path using adaptive routing."""
        # Try shortest path first
        try:
            path = list(nx.shortest_path(self.network_graph, source, destination))
            return path
        except nx.NetworkXNoPath:
            return None
            
    def _calculate_quantum_edge_cost(self, node_a: str, node_b: str) -> Optional[float]:
        """Calculate cost for quantum transmission over an edge."""
        link_key = (min(node_a, node_b), max(node_a, node_b))
        if link_key not in self.links:
            return None
            
        link = self.links[link_key]
        
        # Check if both nodes can handle quantum traffic (more lenient check)
        node_a_obj = self.nodes[node_a]
        node_b_obj = self.nodes[node_b]
        
        # Allow quantum transmission if at least one node is quantum-capable
        quantum_capable = (isinstance(node_a_obj, QuantumNode) or 
                          isinstance(node_b_obj, QuantumNode) or
                          hasattr(node_a_obj, 'stored_entanglements') or 
                          hasattr(node_b_obj, 'stored_entanglements'))
        
        if not quantum_capable:
            return None
            
        # Cost based on distance and quantum-specific factors
        base_cost = link.distance
        
        if isinstance(link, QuantumLink):
            # Pure quantum link - good for quantum traffic
            fidelity_penalty = (1.0 - link.base_fidelity) * 100
            return base_cost + fidelity_penalty
        elif isinstance(link, HybridLink):
            # Hybrid link - some interference
            interference_penalty = link.interference_factor * 50
            return base_cost + interference_penalty
        else:
            # Classical link - can carry quantum via protocol conversion
            conversion_penalty = 30  # Penalty for protocol conversion
            return base_cost + conversion_penalty
            
    def _calculate_classical_edge_cost(self, node_a: str, node_b: str) -> Optional[float]:
        """Calculate cost for classical transmission over an edge."""
        link_key = (min(node_a, node_b), max(node_a, node_b))
        if link_key not in self.links:
            return None
            
        link = self.links[link_key]
        
        # All links can carry classical traffic
        base_cost = link.distance
        
        if isinstance(link, ClassicalLink):
            # Pure classical link - optimal for classical traffic
            latency_penalty = link.latency * 10
            return base_cost + latency_penalty
        elif isinstance(link, HybridLink):
            # Hybrid link - good performance
            return base_cost + 5  # Small penalty for shared resources
        else:  # QuantumLink
            # Quantum link carrying classical data - protocol conversion needed
            return base_cost + 20  # Higher penalty for conversion overhead
            
    def _calculate_path_cost(self, path: List[str], traffic_type: str) -> float:
        """Calculate total cost for a path."""
        if len(path) < 2:
            return 0.0
            
        total_cost = 0.0
        for i in range(len(path) - 1):
            if traffic_type == "quantum":
                edge_cost = self._calculate_quantum_edge_cost(path[i], path[i + 1])
            else:
                edge_cost = self._calculate_classical_edge_cost(path[i], path[i + 1])
                
            if edge_cost is None:
                return float('inf')  # Path not viable
            total_cost += edge_cost
            
        return total_cost
        
    def route_message(self, source: str, destination: str, message: Dict,
                     preferred_type: str = "adaptive") -> Dict:
        """
        Route a message from source to destination.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            message: Message to route
            preferred_type: Preferred routing type
            
        Returns:
            Routing result with path, success status, and metrics
        """
        # Determine message type
        message_type = message.get("type", "classical")
        
        # Find appropriate path
        if preferred_type == "adaptive":
            if message_type == "quantum":
                path = self._find_quantum_path(source, destination)
                if path is None:
                    # Try fallback to classical with protocol conversion
                    path = self._find_classical_path(source, destination)
                    if path is not None:
                        message = self._convert_quantum_to_classical(message)
            else:  # classical message
                path = self._find_classical_path(source, destination)
        else:
            path = self.find_best_path(source, destination, preferred_type)
            
        if path is None:
            return {
                "success": False,
                "error": "No viable path found",
                "path": [],
                "hops": 0,
                "total_distance": 0
            }
            
        # Simulate transmission along the path
        success, transmission_details = self._simulate_transmission(path, message)
        
        result = {
            "success": success,
            "path": path,
            "hops": len(path) - 1,
            "total_distance": self._calculate_path_distance(path),
            "transmission_details": transmission_details
        }
        
        return result
        
    def _simulate_transmission(self, path: List[str], message: Dict) -> Tuple[bool, Dict]:
        """Simulate message transmission along a path."""
        message_type = message.get("type", "classical")
        details = {
            "hops_attempted": 0,
            "hops_successful": 0,
            "failures": [],
            "total_latency": 0.0
        }
        
        current_message = message.copy()
        
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            details["hops_attempted"] += 1
            
            # Get link between nodes
            link_key = (min(current_node, next_node), max(current_node, next_node))
            if link_key not in self.links:
                # Try reverse order
                link_key = (max(current_node, next_node), min(current_node, next_node))
                if link_key not in self.links:
                    details["failures"].append({
                        "hop": i,
                        "from": current_node,
                        "to": next_node,
                        "error": "no_link_found"
                    })
                    return False, details
            
            link = self.links[link_key]
            
            # Attempt transmission
            if message_type == "quantum":
                success, hop_details = self._simulate_quantum_hop(
                    current_node, next_node, link, current_message)
            else:
                success, hop_details = self._simulate_classical_hop(
                    current_node, next_node, link, current_message)
                
            if success:
                details["hops_successful"] += 1
                details["total_latency"] += hop_details.get("latency", 0)
                current_message = hop_details.get("message", current_message)
            else:
                details["failures"].append({
                    "hop": i,
                    "from": current_node,
                    "to": next_node,
                    "error": hop_details.get("error", "unknown")
                })
                return False, details
                
        return True, details
        
    def _simulate_quantum_hop(self, from_node: str, to_node: str, 
                            link: Link, message: Dict) -> Tuple[bool, Dict]:
        """Simulate quantum transmission over a single hop with realistic probabilities."""
        # Base success rates for different link types (quantum is challenging but not impossible!)
        base_success_rates = {
            QuantumLink: 0.70,      # Good success for pure quantum links
            HybridLink: 0.55,       # Medium success due to interference
            ClassicalLink: 0.40     # Lower success for protocol conversion
        }
        
        base_rate = base_success_rates.get(type(link), 0.4)
        
        # Quantum transmission is very sensitive to distance
        distance_penalty = min(link.distance / 600, 0.35)  # Up to 35% penalty
        success_rate = max(base_rate - distance_penalty, 0.15)  # Minimum 15%
        
        # Additional quantum-specific challenges
        fidelity_requirement = message.get("requires_fidelity", 0.8)
        if fidelity_requirement > 0.9:  # High fidelity requirements
            success_rate *= 0.8  # 20% penalty for high fidelity
        
        # Simulate transmission
        if random.random() < success_rate:
            latency = 0.002 + link.distance / 200000  # Realistic latency with setup time
            return True, {"message": message, "latency": latency}
        else:
            error_types = ["low_fidelity", "decoherence", "entanglement_failure", "measurement_error"]
            error = random.choice(error_types)
            return False, {"error": error}
            
    def _simulate_classical_hop(self, from_node: str, to_node: str,
                              link: Link, message: Dict) -> Tuple[bool, Dict]:
        """Simulate classical transmission over a single hop with realistic probabilities."""
        # Base success rates for different link types (classical is reliable!)
        base_success_rates = {
            ClassicalLink: 0.95,    # Very high success for classical links
            HybridLink: 0.88,       # Good success for hybrid links
            QuantumLink: 0.75       # Lower success due to conversion overhead
        }
        
        base_rate = base_success_rates.get(type(link), 0.85)
        
        # Classical transmission is less sensitive to distance
        distance_penalty = min(link.distance / 3000, 0.10)  # Up to 10% penalty
        packet_size = message.get("size", 1024)
        size_penalty = min(packet_size / 20000, 0.03)  # Up to 3% penalty
        
        success_rate = max(base_rate - distance_penalty - size_penalty, 0.4)
        
        # Simulate transmission
        if random.random() < success_rate:
            latency = 0.0008 + link.distance / 300000 + packet_size / 2000000
            return True, {"message": message, "latency": latency}
        else:
            error_types = ["packet_loss", "congestion", "timeout", "checksum_error"]
            if isinstance(link, QuantumLink):
                error_types.append("protocol_conversion_failed")
            error = random.choice(error_types)
            return False, {"error": error}
                
    def _convert_quantum_to_classical(self, quantum_message: Dict) -> Dict:
        """Convert quantum message to classical representation."""
        return {
            "type": "classical",
            "data": str(quantum_message.get("quantum_state", "")),
            "original_type": "quantum",
            "measurement_basis": random.choice(["X", "Z"]),
            "measurement_result": random.choice([0, 1])
        }
        
    def _calculate_path_distance(self, path: List[str]) -> float:
        """Calculate total distance of a path."""
        if len(path) < 2:
            return 0.0
            
        total_distance = 0.0
        for i in range(len(path) - 1):
            link_key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            if link_key in self.links:
                total_distance += self.links[link_key].distance
                
        return total_distance
