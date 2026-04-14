import glm
from typing import List, Dict, Any

class Keyframe:
    """
    Represents a discrete snapshot of an entity's component states at a specific point in time.
    Capable of storing Transform, Mesh, Light, and Camera parameters for interpolation.
    """
    def __init__(self, time: float) -> None:
        self.time: float = time
        self.has_transform: bool = False
        self.position: glm.vec3 = glm.vec3(0.0)
        self.rotation: glm.quat = glm.quat(1.0, 0.0, 0.0, 0.0)
        self.scale: glm.vec3 = glm.vec3(1.0)
        
        self.has_mesh: bool = False
        self.mesh_visible: bool = True
        
        self.has_light: bool = False
        self.light_on: bool = True
        self.light_intensity: float = 1.0
        self.light_color: glm.vec3 = glm.vec3(1.0)
        
    def set_transform(self, pos: glm.vec3, rot: glm.quat, scale: glm.vec3) -> None:
        """Records the spatial configuration of the entity."""
        self.has_transform = True
        self.position = glm.vec3(pos)
        self.rotation = glm.quat(rot)
        self.scale = glm.vec3(scale)
        
    def set_mesh(self, visible: bool) -> None:
        """Records the rendering state of the associated mesh."""
        self.has_mesh = True
        self.mesh_visible = visible
        
    def set_light(self, on: bool, intensity: float, color: glm.vec3) -> None:
        """Records the illumination parameters of the associated light source."""
        self.has_light = True
        self.light_on = on
        self.light_intensity = intensity
        self.light_color = glm.vec3(color)

    def serialize(self) -> Dict[str, Any]:
        """Packages the keyframe data into a JSON-compatible dictionary."""
        data: Dict[str, Any] = {"time": self.time}
        if self.has_transform:
            data["transform"] = {
                "pos": [self.position.x, self.position.y, self.position.z], 
                "rot": [self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z], 
                "scale": [self.scale.x, self.scale.y, self.scale.z]
            }
        if self.has_mesh:
            data["mesh"] = {"visible": self.mesh_visible}
        if self.has_light:
            data["light"] = {
                "on": self.light_on, 
                "intensity": self.light_intensity, 
                "color": [self.light_color.x, self.light_color.y, self.light_color.z]
            }
        return data

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'Keyframe':
        """Reconstructs a Keyframe instance from a JSON payload."""
        kf = Keyframe(data.get("time", 0.0))
        if "transform" in data:
            t_data = data["transform"]
            kf.set_transform(
                glm.vec3(*t_data.get("pos", [0.0, 0.0, 0.0])), 
                glm.quat(*t_data.get("rot", [1.0, 0.0, 0.0, 0.0])), 
                glm.vec3(*t_data.get("scale", [1.0, 1.0, 1.0]))
            )
        if "mesh" in data:
            kf.set_mesh(data["mesh"].get("visible", True))
        if "light" in data:
            l_data = data["light"]
            kf.set_light(
                l_data.get("on", True), 
                l_data.get("intensity", 1.0), 
                glm.vec3(*l_data.get("color", [1.0, 1.0, 1.0]))
            )
        return kf


class AnimationComponent:
    """
    Stores keyframe tracks and kinematic properties for dynamic entities.
    Acts as the central timeline for entity-specific state changes over time.
    """
    def __init__(self) -> None:
        self.is_active: bool = True
        self.keyframes: List[Keyframe] = []
        self.current_time: float = 0.0
        self.duration: float = 0.0
        self.loop: bool = False
        
        # UI Tracking State for Auto-Keying workflow
        self.active_keyframe_index: int = -1 
        
        # Legacy Kinematic state
        self.velocity: glm.vec3 = glm.vec3(0.0)
        self.angular_velocity: glm.vec3 = glm.vec3(0.0)
        
        # Initialize anchor keyframe
        self.add_keyframe(Keyframe(0.0))

    def add_keyframe(self, keyframe: Keyframe) -> None:
        self.keyframes.append(keyframe)
        self._sort_and_update_duration()

    def remove_keyframe(self, index: int) -> None:
        if 0 <= index < len(self.keyframes):
            self.keyframes.pop(index)
            self.active_keyframe_index = -1 
            self._sort_and_update_duration()

    def _sort_and_update_duration(self) -> None:
        if not self.keyframes:
            self.duration = 0.0
            return
        self.keyframes.sort(key=lambda k: k.time)
        self.duration = self.keyframes[-1].time

    def serialize(self) -> Dict[str, Any]:
        return {
            "is_active": self.is_active,
            "current_time": self.current_time,
            "loop": self.loop,
            "keyframes": [k.serialize() for k in self.keyframes],
            "vel": [self.velocity.x, self.velocity.y, self.velocity.z],
            "ang_vel": [self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z]
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        self.is_active = data.get("is_active", True)
        self.current_time = data.get("current_time", 0.0)
        self.loop = data.get("loop", False)
        self.active_keyframe_index = -1
        
        kf_data = data.get("keyframes", [])
        if kf_data:
            self.keyframes = [Keyframe.deserialize(k) for k in kf_data]
        else:
            self.keyframes = [Keyframe(0.0)]
            
        self._sort_and_update_duration()
        self.velocity = glm.vec3(*data.get("vel", [0.0, 0.0, 0.0]))
        self.angular_velocity = glm.vec3(*data.get("ang_vel", [0.0, 0.0, 0.0]))