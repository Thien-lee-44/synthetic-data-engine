import os
import glm
from typing import Any, Optional
from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.geometry.primitives import PrimitivesManager
from src.engine.resources.resource_manager import ResourceManager
from src.engine.graphics.buffer_objects import BufferObject

from src.engine.scene.components.animation_cmp import AnimationComponent
from src.engine.scene.components.semantic_cmp import SemanticComponent  # [NEW] Import Semantic
from src.engine.synthetic.tracking_mgr import TrackingManager

from src.app.exceptions import SimulationError, ResourceError
from src.app.config import MAX_LIGHTS, DEFAULT_GROUP_NAME, DEFAULT_PROXY_SCALE


class EntityFactory:
    """
    Implements the Abstract Factory design pattern for assembling complex Entity configurations.
    Strictly adheres to the Single Responsibility Principle by decoupling Animation from Semantics.
    """
    
    def __init__(self, scene: Any) -> None:
        self.scene = scene

    def _attach_animation(self, ent: Entity) -> None:
        """Grants the entity the ability to be keyframed over time."""
        ent.add_component(AnimationComponent())

    def _attach_semantic(self, ent: Entity, class_id: int = 0, track_id: Optional[int] = None) -> int:
        """
        Grants the entity Ground Truth tracking data for AI dataset generation.
        Returns the assigned track_id to propagate to child entities (Semantic Inheritance).
        """
        if track_id is None:
            track_id = TrackingManager.get_next_id()
            
        ent.add_component(SemanticComponent(track_id=track_id, class_id=class_id))
        return track_id

    def add_empty_group(self) -> None:
        """Spawns an empty transform node primarily used for hierarchical grouping."""
        ent = Entity(DEFAULT_GROUP_NAME, is_group=True)
        ent.add_component(TransformComponent())
        
        self._attach_animation(ent)
        self._attach_semantic(ent, class_id=3) # Default to Misc (3)
        
        self.scene.add_entity(ent)

    def spawn_primitive(self, name: str, is_2d: bool) -> None:
        """Instantiates an entity equipped with a foundational geometric mesh."""
        geom = PrimitivesManager.get_primitive(name, is_2d)
        
        if geom:
            ent = Entity(name)
            ent.add_component(TransformComponent())
            
            self._attach_animation(ent)
            self._attach_semantic(ent, class_id=3) # Default to Misc (3)
            
            renderer = ent.add_component(MeshRenderer())
            renderer.geometry = geom
            
            if hasattr(geom, 'materials') and 'default_active' in geom.materials:
                renderer.material.setup_from_dict(geom.materials['default_active'])
                
            self.scene.add_entity(ent)

    def spawn_math_surface(self, formula: str, xmin: float, xmax: float, ymin: float, ymax: float, res: int) -> None:
        """Instantiates an entity rendering a procedurally generated mathematical surface."""
        from src.engine.geometry.math_surface import MathSurface
        
        ent = Entity(f"Math: {formula}")
        ent.add_component(TransformComponent())
        
        self._attach_animation(ent)
        self._attach_semantic(ent, class_id=3) # Default to Misc (3)
        
        renderer = ent.add_component(MeshRenderer())
        geom = MathSurface(formula, (xmin, xmax), (ymin, ymax), res)
        geom.formula_str = formula
        renderer.geometry = geom
        
        self.scene.add_entity(ent)

    def add_light(self, light_type: str, proxy_enabled: bool, global_light_on: bool) -> None:
        """Instantiates a light source. Lights are NEVER given a SemanticComponent."""
        current_count = sum(1 for _, l, _ in self.scene.cached_lights if l.type == light_type)
        limit = MAX_LIGHTS.get(light_type, 0)
        
        if current_count >= limit:
            raise SimulationError(f"Cannot add {light_type} Light. Maximum limit reached ({limit} lights).")

        ent = Entity(f"{light_type} Light")
        tf = ent.add_component(TransformComponent())
        
        self._attach_animation(ent) # Lights can move/blink
        # NO _attach_semantic -> Lights won't appear in YOLO datasets
        
        light_comp = ent.add_component(LightComponent(light_type=light_type))
        light_comp.on = global_light_on
        
        if light_type != "Directional":
            renderer = ent.add_component(MeshRenderer())
            renderer.is_proxy = True
            renderer.visible = proxy_enabled
            
            if light_type == "Point": 
                renderer.geometry = PrimitivesManager.get_proxy("proxy_point.ply")
                tf.scale = glm.vec3(DEFAULT_PROXY_SCALE)
            elif light_type == "Spot": 
                renderer.geometry = PrimitivesManager.get_proxy("proxy_spot.ply")
                tf.scale = glm.vec3(DEFAULT_PROXY_SCALE)
                
        self.scene.add_entity(ent)

    def add_camera(self, proxy_enabled: bool) -> None:
        """Instantiates an auxiliary view frustum. Cameras are NEVER given a SemanticComponent."""
        ent = Entity("Camera")
        tf = ent.add_component(TransformComponent())
        
        self._attach_animation(ent) # Cameras can fly/pan
        # NO _attach_semantic -> Cameras won't appear in YOLO datasets
        
        cam = ent.add_component(CameraComponent(mode="Perspective"))
        cam.is_active = not any(e.get_component(CameraComponent) for e in self.scene.entities)
        
        renderer = ent.add_component(MeshRenderer())
        renderer.is_proxy = True
        renderer.visible = proxy_enabled
        tf.scale = glm.vec3(DEFAULT_PROXY_SCALE)
        renderer.geometry = PrimitivesManager.get_proxy("proxy_camera.ply")
        
        self.scene.add_entity(ent)

    def spawn_model_from_path(self, path: str) -> None:
        """Parses a 3D asset file and translates its sub-meshes into ECS entities."""
        try:
            mesh_data_list = ResourceManager.get_model(path)
            raw_name = os.path.splitext(os.path.basename(path))[0]
            display_name = raw_name.replace('_', ' ').title()

            if len(mesh_data_list) > 1:
                parent_ent = Entity(display_name, is_group=True)
                parent_ent.add_component(TransformComponent())
                
                self._attach_animation(parent_ent)
                group_track_id = self._attach_semantic(parent_ent, class_id=0) # Default to Car (0)
                
                self.scene.add_entity(parent_ent)
                
                for sub_data in mesh_data_list:
                    child_ent = Entity(getattr(sub_data, 'name', 'SubMesh'))
                    child_ent.add_component(TransformComponent())
                    
                    self._attach_animation(child_ent)
                    self._attach_semantic(child_ent, class_id=0, track_id=group_track_id)
                    
                    renderer = child_ent.add_component(MeshRenderer())
                    renderer.geometry = BufferObject(sub_data.vertices, sub_data.indices, vertex_size=sub_data.vertex_size)
                    renderer.geometry.filepath = path
                    renderer.geometry.name = getattr(sub_data, 'name', '')
                    
                    if hasattr(sub_data, 'materials') and 'default_active' in sub_data.materials:
                        renderer.material.setup_from_dict(sub_data.materials['default_active'])
                            
                    parent_ent.add_child(child_ent, keep_world=False)
                    self.scene.add_entity(child_ent)
                    
                self.scene.selected_index = self.scene.entities.index(parent_ent)
            else:
                ent = Entity(display_name)
                ent.add_component(TransformComponent())
                
                self._attach_animation(ent)
                self._attach_semantic(ent, class_id=0)
                
                renderer = ent.add_component(MeshRenderer())
                sub_data = mesh_data_list[0]
                
                renderer.geometry = BufferObject(sub_data.vertices, sub_data.indices, vertex_size=sub_data.vertex_size)
                renderer.geometry.filepath = path
                renderer.geometry.name = getattr(sub_data, 'name', '')
                
                if hasattr(sub_data, 'materials') and 'default_active' in sub_data.materials:
                    renderer.material.setup_from_dict(sub_data.materials['default_active'])
                        
                self.scene.add_entity(ent)
                
        except Exception as e:
            raise ResourceError(f"Failed to instantiate model hierarchy from '{path}'.\nReason: {e}")