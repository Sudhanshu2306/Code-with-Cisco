from quantum_network.nodes import ClassicalNode, QuantumNode
import random
import networkx as nx
class QuantumRepeaterNode(QuantumNode):
    def __init__(self, node_id, max_qubits=4, decoherence_rate=0.2):
        super().__init__(node_id, max_qubits, decoherence_rate)
        self.is_repeater = True

def build_sample_network_with_repeaters():
    node_ids = ["Q1", "Q2", "Q3", "C1", "C2", "Q4", "Q5", "Q6", "C3", "C4"]

    repeaters = {"Q3", "Q5"}  # Example: promote Q3 and Q5 to repeaters

    G = nx.Graph()
    for nid in node_ids:
        if nid in repeaters:
            G.add_node(nid, nodeobj=QuantumRepeaterNode(nid), ntype="quantum_repeater")
        elif nid.startswith("Q"):
            G.add_node(nid, nodeobj=QuantumNode(nid), ntype="quantum")
        else:
            G.add_node(nid, nodeobj=ClassicalNode(nid), ntype="classical")
        # addition of edges
    return G

def transmit_quantum_packet(sender, receiver):
    rate = sender.decoherence_rate
    if hasattr(sender, "is_repeater") and sender.is_repeater:
        rate = rate * 0.5  # Repeaters halve decoherence
    if random.random() < rate:
        print(f"Quantum {sender.node_id}->{receiver.node_id}: FAIL (decoherence)")
        return False
    if receiver.store_qubit():
        print(f"Quantum {sender.node_id}->{receiver.node_id}: OK")
        return True
    else:
        print(f"Quantum {sender.node_id}->{receiver.node_id}: FAIL (no memory)")
        return False
    