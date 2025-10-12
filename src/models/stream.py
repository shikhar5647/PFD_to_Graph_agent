from typing import Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum
from typing import List

class StreamType(str, Enum):
    """Types of process streams"""
    MATERIAL = "material"
    ENERGY = "energy"
    UTILITY = "utility"
    RECYCLE = "recycle"
    BYPASS = "bypass"

class Stream(BaseModel):
    """Represents a process stream (edge in graph)"""
    id: str = Field(..., description="Unique stream identifier")
    source: str = Field(..., description="Source equipment ID")
    target: str = Field(..., description="Target equipment ID")
    
    # Stream properties
    stream_type: StreamType = Field(default=StreamType.MATERIAL, description="Type of stream")
    label: Optional[str] = Field(None, description="Stream label if any")
    
    # Flow properties (if available)
    properties: Dict[str, any] = Field(default_factory=dict, description="Stream properties")
    phase: Optional[str] = Field(None, description="Phase (liquid/gas/solid)")
    
    # Visual information
    path_points: Optional[List[List[float]]] = Field(None, description="Path coordinates")
    confidence: float = Field(default=1.0, description="Detection confidence")
    
    class Config:
        use_enum_values = True