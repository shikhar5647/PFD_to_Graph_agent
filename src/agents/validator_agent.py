import networkx as nx
from typing import Dict, List

class ValidatorAgent:
    """Agent for validating the constructed graph"""
    
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config
    
    def validate(self, graph: nx.DiGraph, graph_data: Dict) -> Dict:
        """Validate the constructed graph"""
        errors = []
        warnings = []
        
        # Check if graph is connected
        if not nx.is_weakly_connected(graph):
            warnings.append("Graph has disconnected components")
        
        # Check for isolated nodes
        isolated = list(nx.isolates(graph))
        if isolated:
            warnings.append(f"Found {len(isolated)} isolated nodes: {isolated}")
        
        # Check for cycles (acceptable in recycle streams)
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            warnings.append(f"Found {len(cycles)} cycles (may be recycle streams)")
        
        # Check node degrees
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            if in_degree == 0 and out_degree == 0:
                errors.append(f"Node {node} has no connections")
        
        # Validate equipment types
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'unknown':
                warnings.append(f"Node {node} has unknown equipment type")
        
        passed = len(errors) == 0
        
        return {
            "passed": passed,
            "errors": errors,
            "warnings": warnings
        }
    
    def refine(self, graph: nx.DiGraph, errors: List[str]) -> Dict:
        """Attempt to fix validation errors"""
        # Implement refinement logic
        # For now, return original
        return {
            "refined_nodes": [],
            "refined_edges": []
        }