import networkx as nx
from typing import Dict, List
from ..models.pfd_graph import PFDGraph
from ..models.equipment import Equipment, EquipmentType
from ..models.stream import Stream, StreamType

class GraphBuilderAgent:
    """Agent for constructing the final NetworkX graph"""
    
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config
    
    def build_graph(
        self,
        symbols: List[Dict],
        labels: Dict[str, str],
        connections: List[Dict]
    ) -> Dict:
        """Build NetworkX graph from detected elements"""
        equipment_list = []
        stream_list = []
        
        # Create equipment nodes
        for idx, symbol in enumerate(symbols):
            symbol_id = f"eq_{idx}"
            label = labels.get(symbol_id, f"unlabeled_{idx}")
            
            equipment = Equipment(
                id=symbol_id,
                label=label,
                equipment_type=self._infer_equipment_type(symbol, label),
                bbox=symbol.get("bbox"),
                center=symbol.get("center"),
                confidence=symbol.get("confidence", 1.0)
            )
            equipment_list.append(equipment)
        
        # Create stream edges
        for idx, conn in enumerate(connections):
            stream = Stream(
                id=f"stream_{idx}",
                source=conn["source"],
                target=conn["target"],
                stream_type=StreamType.MATERIAL,
                path_points=conn.get("path"),
                confidence=conn.get("confidence", 1.0)
            )
            stream_list.append(stream)
        
        # Create PFDGraph object
        pfd_graph = PFDGraph(
            equipment_list=equipment_list,
            stream_list=stream_list
        )
        
        # Convert to NetworkX
        nx_graph = pfd_graph.to_networkx()
        
        return {
            "nodes": [e.dict() for e in equipment_list],
            "edges": [s.dict() for s in stream_list],
            "graph_dict": pfd_graph.dict(),
            "networkx": nx_graph
        }
    
    def _infer_equipment_type(self, symbol: Dict, label: str) -> EquipmentType:
        """Infer equipment type from symbol and label"""
        label_lower = label.lower()
        
        if any(x in label_lower for x in ['reactor', 'r-', 'rx']):
            return EquipmentType.REACTOR
        elif any(x in label_lower for x in ['hex', 'heat', 'exchanger']):
            return EquipmentType.HEAT_EXCHANGER
        elif any(x in label_lower for x in ['dist', 'column', 't-']):
            return EquipmentType.DISTILLATION
        elif 'mix' in label_lower:
            return EquipmentType.MIXER
        elif 'splt' in label_lower or 'split' in label_lower:
            return EquipmentType.SPLITTER
        elif 'pump' in label_lower or 'pp' in label_lower:
            return EquipmentType.PUMP
        elif any(x in label_lower for x in ['valve', 'v-']):
            return EquipmentType.VALVE
        elif 'raw' in label_lower:
            return EquipmentType.RAW_MATERIAL
        elif 'prod' in label_lower:
            return EquipmentType.PRODUCT
        else:
            return EquipmentType.UNKNOWN