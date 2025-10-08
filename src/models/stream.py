from typing import Optional, Dict
from pydantic import BaseModel, Field

class StreamType(str, Enum):
    """Types of process streams"""
    MATERIAL = "material"
    ENERGY = "energy"
    UTILITY = "utility"
    RECYCLE = "recycle"
    BYPASS = "bypass"