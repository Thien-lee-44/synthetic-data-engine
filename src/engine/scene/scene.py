import os
import re
import glm
from typing import List, Tuple, Any

from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.geometry.primitives import PrimitivesManager
from src.engine.resources.resource_manager import ResourceManager
from src.engine.graphics.buffer_objects import BufferObject

from src.engine.scene.components.animation_cmp import AnimationComponent
from src.engine.scene.components.semantic_cmp import SemanticComponent  # [NEW] Import Semantic
from src.engine.synthetic.tracking_mgr import TrackingManager

from src.app.config import (
    DEFAULT_MANIPULATION_MODE, 
    DEFAULT_CAMERA_NAME, 
    DEFAULT_SCENE_CAM_POS, 
    DEFAULT_PROXY_SCALE, 
    DEFAULT_SCENE_LIGHT_ROT
)


class Scene:
    """
    Acts as the primary state container for the 3D world context, orchestrating all active entities.
    Employs Data-Oriented Design principles (Structural Caching) to optimize the render loop.
    """
    
    def __init__(self) -> None:
        self.entities: List[Entity] = []
        self.selected_index: int = -1 
        self.manipulation_mode: str = DEFAULT_MANIPULATION_MODE
        self.show_screen_axis: bool = True
        
        # Render caches for rapid access during the rendering loop
        self.cached_cameras: List[Tuple[TransformComponent, CameraComponent, Entity]] = []
        self.cached_lights: List[Tuple[TransformComponent, LightComponent, Entity]] = []
        self.cached_renderables: List[Tuple[TransformComponent, MeshRenderer, Entity]] = []
        
        # Bootstrap default scene environment
        self.setup_default_camera()
        self.setup_default_light()
        
        # Bootstrap default geometric entity
        cube_entity = Entity("Default Cube")
        cube_entity.add_component(TransformComponent())
        
        # [CRITICAL FIX] Strictly decouple Animation from Semantics for the Default Cube
        cube_entity.add_component(AnimationComponent())
        cube_entity.add_component(SemanticComponent(track_id=TrackingManager.get_next_id(), class_id=3)) # 3 = Misc
            
        renderer = cube_entity.add_component(MeshRenderer())
        geom = PrimitivesManager.get_primitive("Cube")
        
        if geom: 
            renderer.geometry = geom
            
        self.add_entity(cube_entity)

    def _rebuild_cache(self) -> None:
        """
        Categorizes and explicitly caches active components. 
        This is a deterministic function triggered exclusively whenever the scene topology mutates.
        """
        self.cached_cameras.clear()
        self.cached_lights.clear()
        self.cached_renderables.clear()
        
        for ent in self.entities:
            tf = ent.get_component(TransformComponent)
            if not tf: 
                continue
            
            cam = ent.get_component(CameraComponent)
            if cam: 
                self.cached_cameras.append((tf, cam, ent))
            
            light = ent.get_component(LightComponent)
            if light: 
                self.cached_lights.append((tf, light, ent))
            
            mesh = ent.get_component(MeshRenderer)
            if mesh: 
                self.cached_renderables.append((tf, mesh, ent))

    def _get_unique_name(self, desired_name: str) -> str:
        """Ensures entity names remain strictly unique within the hierarchy."""
        existing_names = {e.name for e in self.entities}
        if desired_name not in existing_names: 
            return desired_name
        
        base_name = desired_name
        counter = 1
        match = re.match(r"^(.*)\s\((\d+)\)$", desired_name)
        
        if match: 
            base_name = match.group(1)
            counter = int(match.group(2))
            
        while f"{base_name} ({counter})" in existing_names: 
            counter += 1
            
        return f"{base_name} ({counter})"

    def add_entity(self, entity: Entity) -> None:
        """Registers a newly instantiated entity into the global scene registry."""
        entity.name = self._get_unique_name(entity.name)
        self.entities.append(entity)
        self.selected_index = len(self.entities) - 1
        self._rebuild_cache()

    def remove_entity(self, index: int) -> None:
        """Safely executes the deletion lifecycle of an entity, recursively culling all attached children."""
        if 0 <= index < len(self.entities):
            ent = self.entities[index]
            
            if ent.parent: 
                ent.parent.remove_child(ent, keep_world=False)
                
            for child in list(ent.children): 
                if child in self.entities: 
                    self.remove_entity(self.entities.index(child))
                    
            if ent in self.entities: 
                self.entities.remove(ent)
                
            self.selected_index = -1
            self._rebuild_cache()

    def clear_entities(self) -> None:
        """Purges the entire scene state."""
        self.entities.clear()
        self.selected_index = -1
        self._rebuild_cache()

    def setup_default_camera(self) -> None:
        """Bootstraps the mandatory viewing frustum required by the rasterizer."""
        cam = Entity(DEFAULT_CAMERA_NAME)
        tf = cam.add_component(TransformComponent())
        tf.position = glm.vec3(*DEFAULT_SCENE_CAM_POS)
        tf.scale = glm.vec3(DEFAULT_PROXY_SCALE) 
        
        # [CRITICAL FIX] Camera only gets Animation (for movement), NO Semantic
        cam.add_component(AnimationComponent())
        
        cam_comp = CameraComponent(mode="Perspective")
        cam_comp.is_active = True 
        cam.add_component(cam_comp)
        
        renderer = cam.add_component(MeshRenderer())
        renderer.is_proxy = True
        
        proxy_path = PrimitivesManager.get_proxy_path("proxy_camera.ply")
        if os.path.exists(proxy_path):
            mesh_list = ResourceManager.get_model(proxy_path)
            if mesh_list:
                sub = mesh_list[0]
                geom = BufferObject(sub.vertices, sub.indices, sub.vertex_size)
                geom.filepath = proxy_path
                renderer.geometry = geom
                
        self.add_entity(cam)

    def setup_default_light(self) -> None:
        """Bootstraps a fundamental global illumination source."""
        light = Entity("Directional Light")
        tf = light.add_component(TransformComponent())
        
        tf.rotation = glm.vec3(*DEFAULT_SCENE_LIGHT_ROT)
        tf.quat_rot = glm.quat(glm.radians(tf.rotation))
        
        # [CRITICAL FIX] Light only gets Animation (for blinking/moving), NO Semantic
        light.add_component(AnimationComponent())
        
        light.add_component(LightComponent(light_type="Directional"))
        
        self.add_entity(light)