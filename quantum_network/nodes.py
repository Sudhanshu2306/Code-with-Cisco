import random
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

class Node:
    def __init__(self, node_id: str, position: Tuple[float, float], node_type: str):
        self.id = node_id
        self.position = position
        self.type = node_type
        self.connected_nodes: Set[str] = set()
        self.routing_table: Dict[str, str] = {}
        self.active = True
        
    def add_connection(self, node_id: str):
        self.connected_nodes.add(node_id)
        
    def remove_connection(self, node_id: str):
        self.connected_nodes.discard(node_id)
        
    def __str__(self):
        return f"{self.type}Node({self.id})"


class ClassicalNode(Node):
    
    def __init__(self, node_id: str, position: Tuple[float, float]):
        super().__init__(node_id, position, "Classical")
        self.packet_buffer: List[Dict] = []
        self.processing_delay = 0.001 
        self.buffer_size = 1000 
        
    def process_packet(self, packet: Dict) -> bool:
        if len(self.packet_buffer) < self.buffer_size:
            self.packet_buffer.append(packet)
            return True
        return False  
        
    def forward_packet(self, packet: Dict, next_hop: str) -> bool:
        return random.random() > 0.01  


class QuantumNode(Node):
    
    def __init__(self, node_id: str, position: Tuple[float, float], 
                 max_entanglement_storage: int = 10):
        super().__init__(node_id, position, "Quantum")
        self.max_entanglement_storage = max_entanglement_storage
        self.stored_entanglements: Dict[str, int] = {}  
        self.qubit_memory_time = 1.0  
        self.entanglement_fidelity = 0.95
        
    def store_entanglement(self, with_node: str) -> bool:
        total_stored = sum(self.stored_entanglements.values())
        if total_stored < self.max_entanglement_storage:
            if with_node not in self.stored_entanglements:
                self.stored_entanglements[with_node] = 0
            self.stored_entanglements[with_node] += 1
            return True
        return False
        
    def consume_entanglement(self, with_node: str) -> bool:
        if with_node in self.stored_entanglements and self.stored_entanglements[with_node] > 0:
            self.stored_entanglements[with_node] -= 1
            if self.stored_entanglements[with_node] == 0:
                del self.stored_entanglements[with_node]
            return True
        return False
        
    def perform_entanglement_swapping(self, node_a: str, node_b: str) -> bool:
        if (self.consume_entanglement(node_a) and 
            self.consume_entanglement(node_b)):
            return random.random() < 0.8  # 80% success rate
        return False
        
    def quantum_teleportation(self, data: Dict, target_node: str) -> bool:
        if self.consume_entanglement(target_node):
            return random.random() < self.entanglement_fidelity
        return False
