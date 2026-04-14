from typing import Dict, Any

class SemanticComponent:
    """
    Stores Ground Truth semantic data for AI dataset generation (YOLO/COCO).
    Strictly decoupled from Animation to enforce the Single Responsibility Principle.
    """
    def __init__(self, track_id: int = -1, class_id: int = 0) -> None:
        # Unique instance ID used to merge sub-meshes into a single bounding box
        self.track_id: int = track_id
        
        # Category ID (e.g., 0: Car, 1: Pedestrian)
        self.class_id: int = class_id  

    def serialize(self) -> Dict[str, Any]:
        """Packages the semantic data into a JSON-compatible dictionary."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Reconstructs the semantic component state from a JSON payload."""
        self.track_id = data.get("track_id", -1)
        self.class_id = data.get("class_id", 0)