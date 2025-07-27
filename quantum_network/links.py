import random
import numpy as np
from typing import Dict, Optional, Tuple
from .utils import calculate_distance, quantum_decoherence_probability


class Link:
    def __init__(self, node_a: str, node_b: str, link_type: str, 
                 distance: float, bandwidth: float = 1.0):
        self.node_a = node_a
        self.node_b = node_b
        self.type = link_type
        self.distance = distance
        self.bandwidth = bandwidth
        self.active = True
        self.usage_count = 0
        
    def get_other_node(self, node_id: str) -> str:
        if node_id == self.node_a:
            return self.node_b
        elif node_id == self.node_b:
            return self.node_a
        else:
            raise ValueError(f"Node {node_id} not connected to this link")
            
    def __str__(self):
        return f"{self.type}Link({self.node_a} <-> {self.node_b})"


class ClassicalLink(Link):
    def __init__(self, node_a: str, node_b: str, distance: float, 
                 bandwidth: float = 1.0, base_loss_rate: float = 0.01):
        super().__init__(node_a, node_b, "Classical", distance, bandwidth)
        self.base_loss_rate = base_loss_rate
        self.latency = self.calculate_latency()
        self.max_retransmissions = 3
        
    def calculate_latency(self) -> float:
        propagation_delay = self.distance / 200000  
        processing_delay = 0.001  # 1ms
        return propagation_delay + processing_delay
        
    def calculate_packet_loss_rate(self) -> float:
        # Loss increases with distance and usage
        distance_factor = 1 + (self.distance / 1000) * 0.01 
        usage_factor = 1 + (self.usage_count / 1000) * 0.005  
        return self.base_loss_rate * distance_factor * usage_factor
        
    def transmit_packet(self, packet: Dict) -> Tuple[bool, float]:
        """
        Transmit a classical packet through the link.
        Returns (success, actual_latency)
        """
        if not self.active:
            return False, 0.0
            
        self.usage_count += 1
        loss_rate = self.calculate_packet_loss_rate()
        
        if random.random() > loss_rate:
            jitter = random.normalvariate(0, self.latency * 0.1)
            actual_latency = max(0, self.latency + jitter)
            return True, actual_latency
        else:
            return False, self.latency
            
    def can_amplify(self) -> bool:
        return True


class QuantumLink(Link):
    def __init__(self, node_a: str, node_b: str, distance: float,
                 base_fidelity: float = 0.95, decoherence_rate: float = 0.1):
        super().__init__(node_a, node_b, "Quantum", distance)
        self.base_fidelity = base_fidelity
        self.decoherence_rate = decoherence_rate
        self.entanglement_generation_rate = 1000  # Hz
        self.max_storage_time = 1.0
        
    def calculate_transmission_fidelity(self) -> float:
        distance_factor = np.exp(-self.distance / 100)  
        return self.base_fidelity * distance_factor
        
    def calculate_decoherence_probability(self, transmission_time: float) -> float:
        return quantum_decoherence_probability(transmission_time, self.decoherence_rate)
        
    def transmit_quantum_state(self, quantum_data: Dict) -> Tuple[bool, Dict]:
        """
        Transmit quantum state through the link.
        Returns (success, result_data)
        """
        if not self.active:
            return False, {}
            
        self.usage_count += 1
        
        transmission_time = self.distance / 299792.458  # km / (km/ms)
        
        # Check for decoherence
        decoherence_prob = self.calculate_decoherence_probability(transmission_time)
        if random.random() < decoherence_prob:
            return False, {"error": "decoherence", "time": transmission_time}
            
        # Check fidelity
        fidelity = self.calculate_transmission_fidelity()
        if random.random() > fidelity:
            return False, {"error": "low_fidelity", "fidelity": fidelity}
            
        # Successful transmission
        result_data = quantum_data.copy()
        result_data["transmission_fidelity"] = fidelity
        result_data["transmission_time"] = transmission_time
        
        return True, result_data
        
    def establish_entanglement(self) -> Tuple[bool, float]:
        """
        Attempt to establish entanglement between link endpoints.
        Returns (success, fidelity)
        """
        if not self.active:
            return False, 0.0
            
        success_prob = np.exp(-self.distance / 50)  
        
        if random.random() < success_prob:
            fidelity = self.calculate_transmission_fidelity()
            return True, fidelity
        else:
            return False, 0.0
            
    def can_amplify(self) -> bool:
        return False
        
    def attempt_amplification(self) -> bool:
        return False
        
    def supports_entanglement_swapping(self) -> bool:
        return True
        
    def measure_bell_state(self) -> Tuple[int, int]:
        return random.randint(0, 1), random.randint(0, 1)


class HybridLink(Link):
    def __init__(self, node_a: str, node_b: str, distance: float,
                 classical_bandwidth: float = 1.0, quantum_fidelity: float = 0.9):
        super().__init__(node_a, node_b, "Hybrid", distance)
        
        # Classical link properties
        self.classical_component = ClassicalLink(node_a, node_b, distance, classical_bandwidth)
        self.quantum_component = QuantumLink(node_a, node_b, distance, quantum_fidelity * 0.9)
        
        self.interference_factor = 0.1
        
    def transmit_classical_packet(self, packet: Dict) -> Tuple[bool, float]:
        success, latency = self.classical_component.transmit_packet(packet)
        
        if random.random() < self.interference_factor:
            success = False
            
        return success, latency
        
    def transmit_quantum_state(self, quantum_data: Dict) -> Tuple[bool, Dict]:
        success, result = self.quantum_component.transmit_quantum_state(quantum_data)
        
        # Check for classical interference
        if success and random.random() < self.interference_factor:
            success = False
            result = {"error": "classical_interference"}
            
        return success, result
        
    def establish_entanglement(self) -> Tuple[bool, float]:
        success, fidelity = self.quantum_component.establish_entanglement()
        
        if success:
            fidelity *= (1 - self.interference_factor)
            
        return success, fidelity
