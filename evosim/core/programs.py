"""
Program representations for the EvoSimC evolutionary system.

This module contains the core data structures representing APL programs
and their metadata throughout the evolutionary process.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class APLProgram:
    """Represents a single APL program with its metadata."""
    apl_code: str
    dps_score: float
    generation: int
    timestamp: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    diff_applied: Optional[str] = None # The diff that was applied to create this program
    reasoning: Optional[str] = None # Place for the program to explain its reasoning
    evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    program_id: str = field(default="")
    
    def __post_init__(self):
        if not self.program_id:
            # Generate a simple ID based on timestamp and hash
            self.program_id = f"apl_{int(self.timestamp)}_{hash(self.apl_code) % 10000:04d}"
