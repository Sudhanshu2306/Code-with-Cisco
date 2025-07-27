#!/usr/bin/env python3
"""
Part 6: Design a Simple Key PKI without Public Keys

In 2030, quantum computers have broken public key cryptography. This script
implements a post-quantum symmetric key distribution system for secure
communication among 25 people, where any two can communicate securely
without the other 23 being able to eavesdrop.

Usage:
    python part6_symmetric_key_system.py
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import hashlib
import random
import secrets
from typing import Dict, List, Set, Tuple, Optional
import os
import itertools
import time


class SymmetricKeyNode:
    """Represents a participant in the symmetric key distribution system."""
    
    def __init__(self, node_id: str, name: str):
        self.id = node_id
        self.name = name
        self.master_secret = secrets.token_bytes(32)  # 256-bit master secret
        self.pairwise_keys = {}  # {other_node_id: shared_key}
        self.key_server_keys = {}  # Keys shared with key servers
        self.messages_sent = 0
        self.messages_received = 0
        
    def derive_key(self, other_node_id: str, shared_secret: bytes) -> bytes:
        """Derive a pairwise key using key derivation function."""
        # Use HKDF-like key derivation
        context = f"{self.id}:{other_node_id}".encode()
        key_material = shared_secret + context
        return hashlib.sha256(key_material).digest()
        
    def encrypt_message(self, message: str, recipient_id: str) -> Optional[bytes]:
        """Encrypt a message for a specific recipient."""
        if recipient_id not in self.pairwise_keys:
            return None
            
        key = self.pairwise_keys[recipient_id]
        
        # Simple XOR encryption for demonstration (use AES in practice)
        message_bytes = message.encode('utf-8')
        nonce = secrets.token_bytes(16)
        
        # XOR with key-derived stream
        key_stream = self._generate_key_stream(key, nonce, len(message_bytes))
        ciphertext = bytes(m ^ k for m, k in zip(message_bytes, key_stream))
        
        self.messages_sent += 1
        return nonce + ciphertext
        
    def decrypt_message(self, ciphertext: bytes, sender_id: str) -> Optional[str]:
        """Decrypt a message from a specific sender."""
        if sender_id not in self.pairwise_keys:
            return None
            
        key = self.pairwise_keys[sender_id]
        
        # Extract nonce and ciphertext
        nonce = ciphertext[:16]
        encrypted_data = ciphertext[16:]
        
        # XOR with key-derived stream
        key_stream = self._generate_key_stream(key, nonce, len(encrypted_data))
        plaintext_bytes = bytes(c ^ k for c, k in zip(encrypted_data, key_stream))
        
        self.messages_received += 1
        return plaintext_bytes.decode('utf-8')
        
    def _generate_key_stream(self, key: bytes, nonce: bytes, length: int) -> bytes:
        """Generate a key stream for encryption/decryption."""
        # Simple key stream generation (use ChaCha20 or AES-CTR in practice)
        stream = b''
        counter = 0
        
        while len(stream) < length:
            block_input = key + nonce + counter.to_bytes(4, 'big')
            block = hashlib.sha256(block_input).digest()
            stream += block
            counter += 1
            
        return stream[:length]


class KeyDistributionCenter:
    """Trusted Key Distribution Center for initial key establishment."""
    
    def __init__(self, kdc_id: str):
        self.id = kdc_id
        self.master_key = secrets.token_bytes(32)
        self.node_keys = {}  # {node_id: shared_key_with_node}
        self.key_requests = 0
        
    def register_node(self, node: SymmetricKeyNode) -> bytes:
        """Register a node and establish shared key."""
        shared_key = secrets.token_bytes(32)
        self.node_keys[node.id] = shared_key
        node.key_server_keys[self.id] = shared_key
        return shared_key
        
    def generate_session_key(self, node_a_id: str, node_b_id: str) -> Tuple[bytes, bytes]:
        """Generate session key for two nodes and create encrypted tickets."""
        if node_a_id not in self.node_keys or node_b_id not in self.node_keys:
            raise ValueError("One or both nodes not registered")
            
        session_key = secrets.token_bytes(32)
        self.key_requests += 1
        
        # Create tickets encrypted with each node's KDC key
        ticket_a = self._create_ticket(session_key, node_b_id, self.node_keys[node_a_id])
        ticket_b = self._create_ticket(session_key, node_a_id, self.node_keys[node_b_id])
        
        return ticket_a, ticket_b
        
    def _create_ticket(self, session_key: bytes, other_party: str, node_key: bytes) -> bytes:
        """Create an encrypted ticket containing session key."""
        # Ticket format: session_key + other_party_id + timestamp
        timestamp = int(time.time()).to_bytes(8, 'big')
        ticket_data = session_key + other_party.encode() + timestamp
        
        # Encrypt with node's KDC key
        nonce = secrets.token_bytes(16)
        key_stream = self._encrypt_ticket_data(ticket_data, node_key, nonce)
        return nonce + key_stream
        
    def _encrypt_ticket_data(self, data: bytes, key: bytes, nonce: bytes) -> bytes:
        """Encrypt ticket data."""
        key_stream = b''
        counter = 0
        
        while len(key_stream) < len(data):
            block_input = key + nonce + counter.to_bytes(4, 'big')
            block = hashlib.sha256(block_input).digest()
            key_stream += block
            counter += 1
            
        return bytes(d ^ k for d, k in zip(data, key_stream[:len(data)]))


class SymmetricKeySystem:
    """
    Post-quantum symmetric key distribution system with multiple approaches.
    """
    
    def __init__(self, num_participants: int = 25):
        self.num_participants = num_participants
        self.participants = {}
        self.approaches = {}
        
    def create_participants(self):
        """Create all participants in the system."""
        print(f"Creating {self.num_participants} participants...")
        
        for i in range(self.num_participants):
            node_id = f"P{i:02d}"
            name = f"Person_{i+1}"
            participant = SymmetricKeyNode(node_id, name)
            self.participants[node_id] = participant
            
        print(f"Created {len(self.participants)} participants")
        
    def implement_approach_1_kdc(self):
        """
        Approach 1: Centralized Key Distribution Center (KDC)
        
        Pros: Simple, efficient key distribution
        Cons: Single point of failure, requires trusted third party
        """
        print("Implementing Approach 1: Centralized KDC...")
        
        # Create KDC
        kdc = KeyDistributionCenter("KDC_01")
        
        # Register all participants with KDC
        for participant in self.participants.values():
            kdc.register_node(participant)
            
        # Generate pairwise keys on-demand
        key_distribution_time = 0
        total_keys = 0
        
        for i, participant_a in enumerate(self.participants.values()):
            for j, participant_b in enumerate(list(self.participants.values())[i+1:], i+1):
                start_time = time.time()
                
                # Request session key from KDC
                ticket_a, ticket_b = kdc.generate_session_key(participant_a.id, participant_b.id)
                
                # Both participants decrypt their tickets to get session key
                session_key = self._extract_session_key_from_ticket(
                    ticket_a, participant_a, kdc
                )
                
                # Store pairwise keys
                participant_a.pairwise_keys[participant_b.id] = session_key
                participant_b.pairwise_keys[participant_a.id] = session_key
                
                key_distribution_time += time.time() - start_time
                total_keys += 1
        
        self.approaches["kdc"] = {
            "name": "Centralized KDC",
            "key_distribution_time": key_distribution_time,
            "total_keys": total_keys,
            "storage_per_node": 1,  # Only need key with KDC
            "communication_rounds": 1,  # One request to KDC
            "trust_requirements": "Trusted KDC",
            "scalability": "O(1) per key request",
            "single_point_failure": True,
            "kdc": kdc
        }
        
        print(f"KDC approach: {total_keys} pairwise keys established")
        
    def _extract_session_key_from_ticket(self, ticket: bytes, 
                                       participant: SymmetricKeyNode,
                                       kdc: KeyDistributionCenter) -> bytes:
        """Extract session key from encrypted ticket."""
        # Decrypt ticket using participant's KDC key
        kdc_key = participant.key_server_keys[kdc.id]
        nonce = ticket[:16]
        encrypted_data = ticket[16:]
        
        # Decrypt
        key_stream = kdc._encrypt_ticket_data(encrypted_data, kdc_key, nonce)
        decrypted_data = bytes(e ^ k for e, k in zip(encrypted_data, key_stream))
        
        # Extract session key (first 32 bytes)
        session_key = decrypted_data[:32]
        return session_key
        
    def implement_approach_2_hierarchical(self):
        """
        Approach 2: Hierarchical Key Distribution
        
        Pros: Reduced single point of failure, better scalability
        Cons: More complex, multiple trust relationships
        """
        print("Implementing Approach 2: Hierarchical Distribution...")
        
        # Create regional KDCs
        num_regions = 5
        participants_per_region = self.num_participants // num_regions
        regional_kdcs = {}
        central_kdc = KeyDistributionCenter("Central_KDC")
        
        # Create regional KDCs and register with central KDC
        for region in range(num_regions):
            region_id = f"Region_{region}"
            regional_kdc = KeyDistributionCenter(f"Regional_KDC_{region}")
            regional_kdcs[region_id] = regional_kdc
            
            # Register regional KDC with central KDC
            central_kdc.node_keys[regional_kdc.id] = secrets.token_bytes(32)
        
        # Assign participants to regions
        participant_list = list(self.participants.values())
        key_distribution_time = 0
        total_keys = 0
        inter_region_keys = 0
        
        # First pass: register all participants with regional and central KDCs
        for i, participant in enumerate(participant_list):
            region = i // participants_per_region
            region = min(region, num_regions - 1)  # Handle remainder
            
            # Register with regional KDC
            regional_kdcs[f"Region_{region}"].register_node(participant)
            # Also register with central KDC for inter-region communication
            central_kdc.register_node(participant)
        
        for i, participant_a in enumerate(participant_list):
            region_a = i // participants_per_region
            region_a = min(region_a, num_regions - 1)
            
            for j, participant_b in enumerate(participant_list[i+1:], i+1):
                region_b = j // participants_per_region
                region_b = min(region_b, num_regions - 1)
                
                start_time = time.time()
                
                if region_a == region_b:
                    # Intra-region communication - use regional KDC
                    regional_kdc = regional_kdcs[f"Region_{region_a}"]
                    ticket_a, ticket_b = regional_kdc.generate_session_key(
                        participant_a.id, participant_b.id
                    )
                else:
                    # Inter-region communication - use central KDC
                    ticket_a, ticket_b = central_kdc.generate_session_key(
                        participant_a.id, participant_b.id
                    )
                    inter_region_keys += 1
                
                # Extract session key
                session_key = secrets.token_bytes(32)  # Simplified for demo
                participant_a.pairwise_keys[participant_b.id] = session_key
                participant_b.pairwise_keys[participant_a.id] = session_key
                
                key_distribution_time += time.time() - start_time
                total_keys += 1
        
        self.approaches["hierarchical"] = {
            "name": "Hierarchical Distribution",
            "key_distribution_time": key_distribution_time,
            "total_keys": total_keys,
            "storage_per_node": 2,  # Regional + Central KDC keys
            "communication_rounds": 1,
            "trust_requirements": "Multiple KDCs",
            "scalability": "O(log n)",
            "single_point_failure": False,
            "inter_region_keys": inter_region_keys,
            "regions": num_regions
        }
        
        print(f"Hierarchical approach: {total_keys} keys, {inter_region_keys} inter-region")
        
    # def implement_approach_3_preshared(self):
    #     """
    #     Approach 3: Pre-shared Key Pools
        
    #     Pros: No online key distribution, perfect forward secrecy possible
    #     Cons: High storage requirements, key exhaustion
    #     """
    #     print("Implementing Approach 3: Pre-shared Key Pools...")
        
    #     # Calculate storage requirements
    #     keys_per_pair = 100  # Pool of 100 keys per pair
    #     total_pairs = (self.num_participants * (self.num_participants - 1)) // 2
    #     total_storage = total_pairs * keys_per_pair * 32  # 32 bytes per key
        
    #     key_distribution_time = 0
    #     start_time = time.time()
        
    #     # Generate key pools for each pair
    #     participant_list = list(self.participants.values())
    #     for i, participant_a in enumerate(participant_list):
    #         for j, participant_b in enumerate(participant_list[i+1:], i+1):
    #             # Generate shared key pool
    #             key_pool = [secrets.token_bytes(32) for _ in range(keys_per_pair)]
                
    #             # Both participants store the same key pool
    #             pair_id = f"{participant_a.id}:{participant_b.id}"
    #             participant_a.pairwise_keys[participant_b.id] = key_pool[0]  # Use first key
    #             participant_b.pairwise_keys[participant_a.id] = key_pool[0]
        
    #     key_distribution_time = time.time() - start_time
        
    #     self.approaches["preshared"] = {
    #         "name": "Pre-shared Key Pools",
    #         "key_distribution_time": key_distribution_time,
    #         "total_keys": total_pairs,
    #         "storage_per_node": (self.num_participants - 1) * keys_per_pair * 32,  # bytes
    #         "communication_rounds": 0,  # No online communication needed
    #         "trust_requirements": "Secure key distribution channel",
    #         "scalability": "O(n²) storage",
    #         "single_point_failure": False,
    #         "total_storage_bytes": total_storage,
    #         "keys_per_pair": keys_per_pair
    #     }
        
    #     print(f"Pre-shared approach: {total_pairs} key pools created")
        
    # def implement_approach_4_threshold(self):
    #     """
    #     Approach 4: Threshold Secret Sharing
        
    #     Pros: No single point of failure, fault tolerant
    #     Cons: Complex coordination, higher overhead
    #     """
    #     print("Implementing Approach 4: Threshold Secret Sharing...")
        
    #     # Create multiple key servers
    #     num_servers = 7
    #     threshold = 4  # Need 4 out of 7 servers
        
    #     key_servers = []
    #     for i in range(num_servers):
    #         server = KeyDistributionCenter(f"KeyServer_{i}")
    #         key_servers.append(server)
        
    #     # Register participants with all servers
    #     for participant in self.participants.values():
    #         for server in key_servers:
    #             server.register_node(participant)
        
    #     key_distribution_time = 0
    #     total_keys = 0
        
    #     participant_list = list(self.participants.values())
        
    #     for i, participant_a in enumerate(participant_list):
    #         for j, participant_b in enumerate(participant_list[i+1:], i+1):
    #             start_time = time.time()
                
    #             # Generate key shares from threshold servers
    #             selected_servers = random.sample(key_servers, threshold)
    #             key_shares = []
                
    #             for server in selected_servers:
    #                 share = secrets.token_bytes(32)
    #                 key_shares.append(share)
                
    #             # Combine shares to create session key (simplified)
    #             session_key = hashlib.sha256(b''.join(key_shares)).digest()
                
    #             participant_a.pairwise_keys[participant_b.id] = session_key
    #             participant_b.pairwise_keys[participant_a.id] = session_key
                
    #             key_distribution_time += time.time() - start_time
    #             total_keys += 1
        
    #     self.approaches["threshold"] = {
    #         "name": "Threshold Secret Sharing",
    #         "key_distribution_time": key_distribution_time,
    #         "total_keys": total_keys,
    #         "storage_per_node": num_servers,  # Keys with all servers
    #         "communication_rounds": threshold,  # Contact threshold servers
    #         "trust_requirements": f"{threshold} out of {num_servers} servers",
    #         "scalability": "O(k) where k is threshold",
    #         "single_point_failure": False,
    #         "fault_tolerance": num_servers - threshold,
    #         "num_servers": num_servers,
    #         "threshold": threshold
    #     }
        
    #     print(f"Threshold approach: {total_keys} keys with {threshold}/{num_servers} threshold")
        
    def simulate_communication(self, num_messages: int = 100):
        """Simulate secure communication using established keys."""
        print(f"Simulating {num_messages} secure communications...")
        
        results = {}
        participant_list = list(self.participants.values())
        
        for approach_name in self.approaches.keys():
            print(f"  Testing {approach_name} approach...")
            
            successful_messages = 0
            failed_messages = 0
            total_time = 0
            
            for _ in range(num_messages):
                # Select random sender and receiver
                sender, receiver = random.sample(participant_list, 2)
                
                start_time = time.time()
                
                # Create test message
                message = f"Secret message from {sender.name} to {receiver.name}"
                
                # Encrypt message
                encrypted = sender.encrypt_message(message, receiver.id)
                
                if encrypted:
                    # Decrypt message
                    decrypted = receiver.decrypt_message(encrypted, sender.id)
                    
                    if decrypted == message:
                        successful_messages += 1
                    else:
                        failed_messages += 1
                else:
                    failed_messages += 1
                
                total_time += time.time() - start_time
            
            results[approach_name] = {
                "successful_messages": successful_messages,
                "failed_messages": failed_messages,
                "success_rate": successful_messages / num_messages,
                "avg_message_time": total_time / num_messages,
                "total_time": total_time
            }
        
        return results
        
    def compare_approaches(self) -> Dict:
        """Compare all implemented approaches."""
        print("Comparing approaches...")
        
        comparison = {}
        
        for name, approach in self.approaches.items():
            # Calculate efficiency metrics
            keys_per_second = approach["total_keys"] / max(approach["key_distribution_time"], 0.001)
            
            comparison[name] = {
                "name": approach["name"],
                "setup_time": approach["key_distribution_time"],
                "keys_per_second": keys_per_second,
                "storage_efficiency": approach["storage_per_node"],
                "communication_efficiency": approach["communication_rounds"],
                "fault_tolerance": not approach.get("single_point_failure", True),
                "scalability_class": approach["scalability"],
                "trust_model": approach["trust_requirements"]
            }
        
        return comparison
        
    def visualize_approaches(self, save_path: str = "results/symmetric_key_comparison.png"):
        """Visualize comparison of different approaches."""
        print("Creating approach comparison visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Post-Quantum Symmetric Key Distribution Approaches', 
                    fontsize=16, fontweight='bold')
        
        # Prepare data
        approach_names = [approach["name"] for approach in self.approaches.values()]
        setup_times = [approach["key_distribution_time"] for approach in self.approaches.values()]
        storage_reqs = [approach["storage_per_node"] for approach in self.approaches.values()]
        comm_rounds = [approach["communication_rounds"] for approach in self.approaches.values()]
        
        # Plot 1: Setup time comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(approach_names)), setup_times, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_xlabel('Approach')
        ax1.set_ylabel('Setup Time (seconds)')
        ax1.set_title('Key Distribution Setup Time')
        ax1.set_xticks(range(len(approach_names)))
        ax1.set_xticklabels([name.split()[0] for name in approach_names], rotation=45)
        
        # Add value labels
        for bar, time_val in zip(bars1, setup_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Plot 2: Storage requirements
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(approach_names)), storage_reqs,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_xlabel('Approach')
        ax2.set_ylabel('Storage per Node')
        ax2.set_title('Storage Requirements')
        ax2.set_xticks(range(len(approach_names)))
        ax2.set_xticklabels([name.split()[0] for name in approach_names], rotation=45)
        
        # Handle different units for pre-shared approach
        for i, (bar, storage) in enumerate(zip(bars2, storage_reqs)):
            height = bar.get_height()
            if i == 2:  # Pre-shared approach
                label = f'{storage/1024:.0f}KB'
            else:
                label = f'{storage}'
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    label, ha='center', va='bottom')
        
        # Plot 3: Communication rounds
        # ax3 = axes[1, 0]
        # bars3 = ax3.bar(range(len(approach_names)), comm_rounds,
        #                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        # ax3.set_xlabel('Approach')
        # ax3.set_ylabel('Communication Rounds')
        # ax3.set_title('Communication Efficiency')
        # ax3.set_xticks(range(len(approach_names)))
        # ax3.set_xticklabels([name.split()[0] for name in approach_names], rotation=45)
        
        # for bar, rounds in zip(bars3, comm_rounds):
        #     height = bar.get_height()
        #     ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
        #             f'{rounds}', ha='center', va='bottom')
        
        # # Plot 4: Scalability and security comparison
        # ax4 = axes[1, 1]
        
        # # Create a qualitative comparison matrix
        # metrics = ['Setup Speed', 'Storage Eff.', 'Comm. Eff.', 'Fault Tolerance', 'Scalability']
        
        # # Scores (1-5 scale)
        # scores = {
        #     'KDC': [5, 5, 5, 1, 4],
        #     'Hierarchical': [4, 4, 4, 3, 4],
        #     'Pre-shared': [2, 1, 5, 5, 2],
        #     'Threshold': [3, 3, 2, 5, 3]
        # }
        
        # x = np.arange(len(metrics))
        # width = 0.2
        
        # for i, (approach, score_list) in enumerate(scores.items()):
        #     ax4.bar(x + i*width, score_list, width, label=approach, alpha=0.8)
        
        # ax4.set_xlabel('Security & Performance Metrics')
        # ax4.set_ylabel('Score (1-5)')
        # ax4.set_title('Qualitative Comparison')
        # ax4.set_xticks(x + width * 1.5)
        # ax4.set_xticklabels(metrics, rotation=45, ha='right')
        # ax4.legend()
        # ax4.set_ylim(0, 6)
        
        # plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Approach comparison saved to {save_path}")
        
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report."""
        
        report = f"""
POST-QUANTUM SYMMETRIC KEY DISTRIBUTION SYSTEM ANALYSIS
======================================================

PROBLEM SCENARIO:
- Year: 2030, quantum computers have broken public key cryptography
- Participants: {self.num_participants} people in a group
- Requirement: Any 2 can communicate securely without others eavesdropping
- Total possible communication pairs: {(self.num_participants * (self.num_participants - 1)) // 2}

IMPLEMENTED APPROACHES:

1. CENTRALIZED KEY DISTRIBUTION CENTER (KDC)
   Description: Single trusted server distributes session keys on demand
   Setup Time: {self.approaches['kdc']['key_distribution_time']:.3f} seconds
   Storage per Node: {self.approaches['kdc']['storage_per_node']} keys
   Communication: {self.approaches['kdc']['communication_rounds']} round per key request
   
   Pros:
   - Simple implementation and management
   - Efficient key distribution
   - Low storage requirements per node
   - Fast key establishment
   
   Cons:
   - Single point of failure
   - Requires continuously trusted third party
   - KDC must be online for all communications
   - Scalability bottleneck
   
   Best for: Small to medium groups with high trust in central authority

2. HIERARCHICAL KEY DISTRIBUTION
   Description: Multiple regional KDCs with central coordination
   Setup Time: {self.approaches['hierarchical']['key_distribution_time']:.3f} seconds
   Storage per Node: {self.approaches['hierarchical']['storage_per_node']} keys
   Regions: {self.approaches['hierarchical']['regions']}
   
   Pros:
   - Reduced single point of failure
   - Better load distribution
   - Improved fault tolerance
   - Scalable architecture
   
   Cons:
   - More complex management
   - Multiple trust relationships required
   - Higher infrastructure costs
   
   Best for: Large organizations with regional structure

3. PRE-SHARED KEY POOLS
   Description: All pairs pre-generate pools of shared keys
   Setup Time: {self.approaches['preshared']['key_distribution_time']:.3f} seconds
   Storage per Node: {self.approaches['preshared']['storage_per_node'] / 1024:.1f} KB
   Keys per Pair: {self.approaches['preshared']['keys_per_pair']}
   
   Pros:
   - No online key distribution needed
   - Perfect forward secrecy possible
   - No single point of failure
   - Immune to network attacks during communication
   
   Cons:
   - Very high storage requirements O(n²)
   - Key exhaustion problem
   - Difficult key refresh
   - Initial distribution challenge
   
   Best for: Small groups with high security requirements, intermittent connectivity

4. THRESHOLD SECRET SHARING
   Description: Multiple servers, require threshold for key generation
   Setup Time: {self.approaches['threshold']['key_distribution_time']:.3f} seconds
   Servers: {self.approaches['threshold']['num_servers']}
   Threshold: {self.approaches['threshold']['threshold']}
   
   Pros:
   - High fault tolerance
   - No single point of failure
   - Distributed trust model
   - Resistant to server compromise
   
   Cons:
   - Complex coordination required
   - Higher communication overhead
   - More expensive infrastructure
   - Slower key establishment
   
   Best for: High-security environments requiring maximum fault tolerance

COMPARATIVE ANALYSIS:

Security Ranking (Best to Worst):
1. Threshold Secret Sharing - Highest security, distributed trust
2. Pre-shared Key Pools - No online vulnerabilities
3. Hierarchical Distribution - Multiple trust points
4. Centralized KDC - Single point of failure

Performance Ranking (Fastest to Slowest):
1. Pre-shared Key Pools - No online computation
2. Centralized KDC - Simple, direct
3. Hierarchical Distribution - Some overhead
4. Threshold Secret Sharing - Complex coordination

Scalability Ranking (Most to Least Scalable):
1. Hierarchical Distribution - O(log n) with regions
2. Centralized KDC - O(1) per request but bottleneck
3. Threshold Secret Sharing - O(k) where k is threshold
4. Pre-shared Key Pools - O(n²) storage

RECOMMENDATIONS:

For the 25-person group scenario:
- If high trust in central authority: Use Centralized KDC
- If moderate security with scalability: Use Hierarchical Distribution  
- If maximum security regardless of cost: Use Threshold Secret Sharing
- If intermittent connectivity: Use Pre-shared Key Pools

Hybrid Approach Recommendation:
Combine approaches for optimal security:
1. Use Hierarchical KDC for routine communications
2. Maintain pre-shared emergency keys for critical situations
3. Implement threshold sharing for high-value key material
4. Regular key rotation using multiple methods

IMPLEMENTATION CONSIDERATIONS:
- All approaches can be enhanced with quantum-resistant algorithms
- Key derivation should use approved post-quantum cryptographic functions
- Regular security audits and key rotation essential
- Consider forward secrecy in all implementations
        """
        
        return report


def main():
    """Main function to demonstrate post-quantum symmetric key systems."""
    print("Quantum-Classical Hybrid Network Simulation")
    print("=" * 50)
    print("Part 6: Design a Simple PKI without Public Keys")
    print()
    
    # Create the system
    key_system = SymmetricKeySystem(num_participants=25)
    
    # Create participants
    key_system.create_participants()
    
    # Implement all approaches
    print("\nImplementing different key distribution approaches...")
    key_system.implement_approach_1_kdc()
    key_system.implement_approach_2_hierarchical()
    # key_system.implement_approach_3_preshared()
    # key_system.implement_approach_4_threshold()
    
    # Simulate communication
    communication_results = key_system.simulate_communication(50)
    
    # Print communication results
    print("\nCommunication Test Results:")
    print("=" * 30)
    for approach, results in communication_results.items():
        print(f"{approach.upper()}: {results['success_rate']:.1%} success rate, "
              f"{results['avg_message_time']*1000:.2f}ms avg time")
    
    # Compare approaches
    comparison = key_system.compare_approaches()
    
    print("\nApproach Comparison:")
    print("=" * 20)
    for name, metrics in comparison.items():
        print(f"\n{metrics['name']}:")
        print(f"  Setup time: {metrics['setup_time']:.3f}s")
        print(f"  Storage efficiency: {metrics['storage_efficiency']}")
        print(f"  Fault tolerant: {metrics['fault_tolerance']}")
        print(f"  Scalability: {metrics['scalability_class']}")
    
    # Visualize results
    key_system.visualize_approaches()
    
    # Generate comprehensive report
    report = key_system.generate_comprehensive_report()
    
    # Save report
    with open("results/symmetric_key_system_report.txt", "w") as f:
        f.write(report)
    
    print("\nPart 6 completed successfully!")
    print("Post-quantum symmetric key distribution system analyzed.")
    print("\nFull report saved to results/symmetric_key_system_report.txt")
    
    return key_system


if __name__ == "__main__":
    key_system = main()
