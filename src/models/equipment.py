# src/models/equipment.py
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class EquipmentType(str, Enum):
    """Standard equipment types in process flowsheets"""
    REACTOR = "reactor"
    HEAT_EXCHANGER = "heat_exchanger"
    DISTILLATION = "distillation_column"
    MIXER = "mixer"
    SPLITTER = "splitter"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    VALVE = "valve"
    TANK = "tank"
    SEPARATOR = "separator"
    RAW_MATERIAL = "raw_material"
    PRODUCT = "product"
    UTILITY = "utility"
    UNKNOWN = "unknown"

class Equipment(BaseModel):
    """Represents a process equipment unit (node in graph)"""
    id: str = Field(..., description="Unique identifier for equipment")
    label: str = Field(..., description="Equipment label from PFD")
    equipment_type: EquipmentType = Field(..., description="Type of equipment")
    
    # Spatial information from image
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x, y, w, h]")
    center: Optional[List[float]] = Field(None, description="Center coordinates [x, y]")
    
    # Process attributes
    properties: Dict[str, Any] = Field(default_factory=dict, description="Equipment properties")
    control_relevant: bool = Field(default=False, description="Key unit for control structure")
    
    # Additional metadata
    confidence: float = Field(default=1.0, description="Detection confidence")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    model_config = ConfigDict(use_enum_values=True)