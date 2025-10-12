# src/agents/graph_builder_agent.py
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
            symbol_id = symbol.get('id', f"eq_{idx}")
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
        
        # Create a set of valid equipment IDs for validation
        valid_equipment_ids = {eq.id for eq in equipment_list}
        
        # Create stream edges - with validation
        for idx, conn in enumerate(connections):
            source = conn.get("source")
            target = conn.get("target")
            
            # Validate that both source and target exist
            if not source or not target:
                print(f"⚠️ Skipping connection {idx}: Missing source or target")
                continue
            
            # Validate that both source and target are valid equipment IDs
            if source not in valid_equipment_ids:
                print(f"⚠️ Skipping connection {idx}: Invalid source '{source}'")
                continue
            
            if target not in valid_equipment_ids:
                print(f"⚠️ Skipping connection {idx}: Invalid target '{target}'")
                continue
            
            # Don't create self-loops
            if source == target:
                print(f"⚠️ Skipping connection {idx}: Self-loop detected ({source} -> {target})")
                continue
            
            try:
                stream = Stream(
                    id=f"stream_{idx}",
                    source=source,
                    target=target,
                    stream_type=StreamType.MATERIAL,
                    path_points=conn.get("path"),
                    confidence=conn.get("confidence", 1.0)
                )
                stream_list.append(stream)
            except Exception as e:
                print(f"⚠️ Error creating stream {idx}: {e}")
                continue
        
        print(f"✅ Created {len(equipment_list)} equipment nodes")
        print(f"✅ Created {len(stream_list)} valid streams (skipped {len(connections) - len(stream_list)} invalid)")
        
        # Create PFDGraph object
        pfd_graph = PFDGraph(
            equipment_list=equipment_list,
            stream_list=stream_list
        )
        
        # Convert to NetworkX
        nx_graph = pfd_graph.to_networkx()
        
        return {
            "nodes": [e.model_dump() for e in equipment_list],
            "edges": [s.model_dump() for s in stream_list],
            "graph_dict": pfd_graph.model_dump(),
            "networkx": nx_graph
        }
    
    def _infer_equipment_type(self, symbol: Dict, label: str) -> EquipmentType:
        """Infer equipment type from symbol and label"""
        label_lower = label.lower()
        
        # Check label keywords
        if any(x in label_lower for x in ['reactor', 'r-', 'rx', 'cstr', 'pfr']):
            return EquipmentType.REACTOR
        elif any(x in label_lower for x in ['hex', 'heat', 'exchanger', 'e-', 'hx']):
            return EquipmentType.HEAT_EXCHANGER
        elif any(x in label_lower for x in ['dist', 'column', 't-', 'tower']):
            return EquipmentType.DISTILLATION
        elif any(x in label_lower for x in ['mix', 'm-']):
            return EquipmentType.MIXER
        elif any(x in label_lower for x in ['splt', 'split', 'splitter']):
            return EquipmentType.SPLITTER
        elif any(x in label_lower for x in ['pump', 'pp', 'p-']):
            return EquipmentType.PUMP
        elif any(x in label_lower for x in ['valve', 'v-', 'cv']):
            return EquipmentType.VALVE
        elif any(x in label_lower for x in ['tank', 'vessel', 'drum']):
            return EquipmentType.TANK
        elif any(x in label_lower for x in ['separator', 'sep', 'flash']):
            return EquipmentType.SEPARATOR
        elif any(x in label_lower for x in ['raw', 'feed']):
            return EquipmentType.RAW_MATERIAL
        elif any(x in label_lower for x in ['prod', 'product', 'output']):
            return EquipmentType.PRODUCT
        
        # Check symbol type from vision detection
        symbol_type = symbol.get('type', '').lower()
        
        if 'circular' in symbol_type or 'circle' in symbol_type:
            return EquipmentType.REACTOR
        elif 'rectangular' in symbol_type or 'rect' in symbol_type:
            return EquipmentType.TANK
        
        return EquipmentType.UNKNOWN