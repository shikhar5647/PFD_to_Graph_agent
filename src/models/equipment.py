from typing import Optional, Dict, List
from pydantic import BaseModel, Field
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