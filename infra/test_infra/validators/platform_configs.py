from dataclasses import dataclass
from typing import Optional


@dataclass
class EC2Config:
    instance_type: str
    node_count: Optional[int] = None


@dataclass
class EKSConfig:
    cluster: str
    namespace: str
