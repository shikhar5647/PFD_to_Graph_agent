import networkx as nx
from typing import List, Dict, Optional,Any
from pydantic import BaseModel, Field
from .equipment import Equipment
from .stream import Stream

class PFDGraph(BaseModel):
    """Complete PFD graph representation"""
    name: Optional[str] = Field(None, description="Flowsheet name")
    equipment_list: List[Equipment] = Field(default_factory=list)
    stream_list: List[Stream] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_image: Optional[str] = Field(None, description="Source image path")
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph"""
        G = nx.DiGraph()
        
        # Add nodes
        for equipment in self.equipment_list:
            G.add_node(
                equipment.id,
                label=equipment.label,
                type=equipment.equipment_type,
                properties=equipment.properties,
                bbox=equipment.bbox,
                center=equipment.center,
                control_relevant=equipment.control_relevant
            )
        
        # Add edges
        for stream in self.stream_list:
            G.add_edge(
                stream.source,
                stream.target,
                stream_id=stream.id,
                label=stream.label,
                stream_type=stream.stream_type,
                properties=stream.properties,
                phase=stream.phase,
                path_points=stream.path_points
            )
        
        # Add graph metadata
        G.graph['name'] = self.name
        G.graph['metadata'] = self.metadata
        
        return G
    
    @classmethod
    def from_networkx(cls, G: nx.DiGraph) -> "PFDGraph":
        """Create PFDGraph from NetworkX graph"""
        equipment_list = []
        for node_id, data in G.nodes(data=True):
            equipment_list.append(Equipment(
                id=node_id,
                label=data.get('label', node_id),
                equipment_type=data.get('type', EquipmentType.UNKNOWN),
                bbox=data.get('bbox'),
                center=data.get('center'),
                properties=data.get('properties', {}),
                control_relevant=data.get('control_relevant', False)
            ))
        
        stream_list = []
        for source, target, data in G.edges(data=True):
            stream_list.append(Stream(
                id=data.get('stream_id', f"{source}_{target}"),
                source=source,
                target=target,
                label=data.get('label'),
                stream_type=data.get('stream_type', StreamType.MATERIAL),
                properties=data.get('properties', {}),
                phase=data.get('phase'),
                path_points=data.get('path_points')
            ))
        
        return cls(
            name=G.graph.get('name'),
            equipment_list=equipment_list,
            stream_list=stream_list,
            metadata=G.graph.get('metadata', {})
        )
    
    class Config:
        arbitrary_types_allowed = True