import os
import math
import json
import copy
import shutil
import random
import numpy as np
import glm
from typing import Dict, List, Any, Optional, Tuple

from src.engine.resources.resource_manager import ResourceManager
from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.scene.components.animation_cmp import AnimationComponent, Keyframe
from src.engine.scene.components.semantic_cmp import SemanticComponent
from src.engine.synthetic.tracking_mgr import TrackingManager

from src.app.exceptions import SimulationError, ResourceError
from src.app.config import MAX_LIGHTS, PASTE_OFFSET, MTL_TOKENS


class SceneManager:
    """
    High-level central controller dictating complex scene operations. 
    Handles logical data routing, hierarchy synchronization, deep serialization 
    (Save/Load/Snapshots), and exporting geometry formats.
    """
    
    _COMPONENT_MAP = {
        "Transform": TransformComponent, 
        "Mesh": MeshRenderer, 
        "Light": LightComponent, 
        "Camera": CameraComponent,
        "Animation": AnimationComponent,
        "Semantic": SemanticComponent
    }

    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.clipboard: Optional[Entity] = None
        
        self.semantic_classes: Dict[int, Dict[str, Any]] = {
            0: {"name": "Car", "color": [1.0, 0.0, 0.0]},           
            1: {"name": "Pedestrian", "color": [0.0, 1.0, 0.0]},    
            2: {"name": "Traffic Sign", "color": [0.0, 0.0, 1.0]},  
            3: {"name": "Misc", "color": [1.0, 1.0, 0.0]}           
        }

    # =========================================================================
    # SEMANTIC CLASS MANAGEMENT
    # =========================================================================

    def get_semantic_classes(self) -> Dict[int, Dict[str, Any]]:
        return self.semantic_classes

    def add_semantic_class(self, name: str) -> int:
        new_id = max(self.semantic_classes.keys()) + 1 if self.semantic_classes else 0
        r, g, b = random.random(), random.random(), random.random()
        self.semantic_classes[new_id] = {"name": name, "color": [r, g, b]}
        return new_id

    def update_semantic_class_color(self, class_id: int, color: List[float]) -> None:
        if class_id in self.semantic_classes:
            self.semantic_classes[class_id]["color"] = color

    # =========================================================================
    # CORE ENTITY MANAGEMENT
    # =========================================================================
    
    def has_clipboard(self) -> bool: 
        return self.clipboard is not None
        
    def get_selected_entity_id(self) -> int: 
        return self.scene.selected_index
        
    def select_entity(self, idx: int) -> None: 
        self.scene.selected_index = idx
        
    def set_manipulation_mode(self, mode: str) -> None: 
        self.scene.manipulation_mode = mode
        
    def get_scene_entities_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": i, 
                "name": e.name, 
                "parent": None if not e.parent else self.scene.entities.index(e.parent), 
                "is_group": e.is_group
            } for i, e in enumerate(self.scene.entities)
        ]

    def reset_entity_transform(self, index: int) -> None:
        if index < 0 or index >= len(self.scene.entities): 
            return
            
        tf = self.scene.entities[index].get_component(TransformComponent)
        if tf:
            tf.position = glm.vec3(0.0)
            tf.rotation = glm.vec3(0.0)
            tf.scale = glm.vec3(1.0)
            tf.quat_rot = glm.quat(glm.vec3(0.0))
            tf.sync_from_gui()

    def update_light_direction(self, yaw: float, pitch: float) -> None:
        if self.scene.selected_index < 0: 
            return
            
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        if tf:
            world_quat = glm.angleAxis(glm.radians(yaw), glm.vec3(0, 1, 0)) * \
                         glm.angleAxis(glm.radians(pitch), glm.vec3(1, 0, 0))
            
            if tf.entity and tf.entity.parent:
                parent_tf = tf.entity.parent.get_component(TransformComponent)
                if parent_tf:
                    tf.quat_rot = glm.inverse(parent_tf.global_quat_rot) * world_quat
                else:
                    tf.quat_rot = world_quat
            else:
                tf.quat_rot = world_quat
                
            tf.rotation = glm.degrees(glm.eulerAngles(tf.quat_rot))
            tf.sync_from_gui()

    def set_active_camera_selected(self) -> None:
        if self.scene.selected_index < 0: 
            return
            
        ent = self.scene.entities[self.scene.selected_index]
        target_cam = ent.get_component(CameraComponent)
        
        if target_cam:
            for e in self.scene.entities:
                c = e.get_component(CameraComponent)
                if c: 
                    c.is_active = False
                    m = e.get_component(MeshRenderer)
                    if m: 
                        m.visible = True
                    
            target_cam.is_active = True
            m_target = ent.get_component(MeshRenderer)
            if m_target: 
                m_target.visible = False

    def get_selected_transform_state(self) -> Optional[Tuple[str, Tuple[float, float, float]]]:
        if self.scene.selected_index < 0: 
            return None
            
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        if not tf: 
            return None
        
        mode = getattr(self.scene, 'manipulation_mode', 'ROTATE')
        val = tf.rotation if mode == "ROTATE" else (tf.position if mode == "MOVE" else tf.scale)
        return (mode, (val.x, val.y, val.z))

    def clear_scene(self) -> None:
        self.scene.entities.clear()
        self.scene.selected_index = -1
        ResourceManager.clear_project_assets()
        self.scene.setup_default_camera()
        self.scene.setup_default_light()

    def get_selected_entity_data(self) -> Optional[Dict[str, Any]]:
        idx = self.scene.selected_index
        if idx < 0 or idx >= len(self.scene.entities): 
            return None
        
        ent = self.scene.entities[idx]
        data = {
            "index": idx, 
            "name": ent.name, 
            "is_group": ent.is_group, 
            "tf": None, 
            "mesh": None, 
            "light": None, 
            "cam": None,
            "anim": None,
            "semantic": None
        }
        
        tf = ent.get_component(TransformComponent)
        if tf: data["tf"] = tf.to_dict()
            
        mesh = ent.get_component(MeshRenderer)
        if mesh: data["mesh"] = mesh.to_dict()
            
        light = ent.get_component(LightComponent)
        if light:
            data["light"] = light.to_dict()
            if tf and light.type in ["Directional", "Spot"]:
                forward = glm.vec3(glm.mat4_cast(tf.global_quat_rot) * glm.vec4(0, 0, -1, 0))
                data["light"]["pitch"] = math.degrees(math.asin(max(-1.0, min(1.0, forward.y))))
                y_val = math.degrees(math.atan2(-forward.x, -forward.z))
                data["light"]["yaw"] = y_val if y_val >= 0 else y_val + 360.0
                
        cam = ent.get_component(CameraComponent)
        if cam: data["cam"] = cam.to_dict()
            
        anim = ent.get_component(AnimationComponent)
        if anim and hasattr(anim, 'serialize'): 
            data["anim"] = anim.serialize()
            
        semantic = ent.get_component(SemanticComponent)
        if semantic and hasattr(semantic, 'serialize'): 
            data["semantic"] = semantic.serialize()
        
        return data

    # =========================================================================
    # COMPONENT PROPERTY MUTATIONS
    # =========================================================================

    def set_component_property(self, comp_name: str, prop: str, value: Any) -> None:
        if self.scene.selected_index < 0: 
            return
            
        ent = self.scene.entities[self.scene.selected_index]
        if comp_name == "Entity": 
            setattr(ent, prop, value)
            return
            
        comp_class = self._COMPONENT_MAP.get(comp_name)
        if not comp_class: 
            return
            
        comp = ent.get_component(comp_class)
        if not comp:
            return

        if comp_name == "Animation":
            self._handle_animation_property(ent, comp, prop, value)
            return

        if comp_name == "Semantic":
            self._handle_semantic_property(ent, comp, prop, value)
            return

        if comp_name == "Transform" and prop in ["position", "rotation", "scale"]:
            setattr(comp, prop, glm.vec3(*value))
            if prop == "rotation": 
                comp.quat_rot = glm.quat(glm.radians(comp.rotation))
            comp.sync_from_gui()
        elif prop.startswith("mat_"): 
            setattr(comp.material, prop[4:], glm.vec3(*value) if isinstance(value, list) else value)
        else: 
            setattr(comp, prop, glm.vec3(*value) if isinstance(value, list) and len(value) == 3 else value)

        # =====================================================================
        # INDEX-BASED AUTO-KEYING SYSTEM
        # Modifying properties while a specific keyframe is explicitly selected
        # will silently overwrite that keyframe regardless of current playhead time.
        # =====================================================================
        anim = ent.get_component(AnimationComponent)
        if anim and hasattr(anim, 'active_keyframe_index') and anim.active_keyframe_index >= 0:
            if anim.active_keyframe_index < len(anim.keyframes):
                kf = anim.keyframes[anim.active_keyframe_index]
                tf = ent.get_component(TransformComponent)
                mesh = ent.get_component(MeshRenderer)
                light = ent.get_component(LightComponent)
                
                if tf: 
                    kf.set_transform(tf.position, tf.quat_rot, tf.scale)
                if mesh: 
                    kf.set_mesh(mesh.visible)
                if light: 
                    kf.set_light(light.on, getattr(light, 'intensity', 1.0), getattr(light, 'color', glm.vec3(1.0)))

    def _handle_animation_property(self, ent: Entity, comp: AnimationComponent, prop: str, value: Any) -> None:
        if prop == "ADD_KEYFRAME":
            target_time = float(value)
            
            # Find existing keyframe within a small epsilon tolerance
            existing_kf = None
            for kf in comp.keyframes:
                if abs(kf.time - target_time) < 0.01:
                    existing_kf = kf
                    break
                    
            kf_to_modify = existing_kf if existing_kf else Keyframe(target_time)
            
            tf = ent.get_component(TransformComponent)
            if tf: 
                kf_to_modify.set_transform(tf.position, tf.quat_rot, tf.scale)
            
            mesh = ent.get_component(MeshRenderer)
            if mesh: 
                kf_to_modify.set_mesh(mesh.visible)
            
            light = ent.get_component(LightComponent)
            if light: 
                intensity = getattr(light, 'intensity', 1.0)
                color = getattr(light, 'color', glm.vec3(1.0))
                kf_to_modify.set_light(light.on, intensity, color)
                
            if not existing_kf:
                comp.add_keyframe(kf_to_modify)
            elif hasattr(comp, '_sort_and_update_duration'):
                # Force internal duration re-calculation just in case
                comp._sort_and_update_duration()
            
        elif prop == "REMOVE_KEYFRAME":
            comp.remove_keyframe(int(value))
            
        elif prop == "CLEAR_KEYFRAMES":
            comp.keyframes.clear()
            comp.duration = 0.0
            
        elif prop == "loop":
            comp.loop = bool(value)

    def _handle_semantic_property(self, ent: Entity, comp: SemanticComponent, prop: str, value: Any) -> None:
        if prop == "class_id":
            comp.class_id = int(value)
            if len(ent.children) > 0:
                self._propagate_semantics(ent, comp.class_id, comp.track_id)
            elif ent.parent is not None:
                comp.track_id = TrackingManager.get_next_id()

    def _propagate_semantics(self, node: Entity, target_class: int, target_track: int) -> None:
        for child in node.children:
            c_sem = child.get_component(SemanticComponent)
            if not c_sem:
                c_sem = child.add_component(SemanticComponent())
                
            c_sem.class_id = target_class
            c_sem.track_id = target_track
            self._propagate_semantics(child, target_class, target_track)

    # =========================================================================
    # HIERARCHY & CLIPBOARD OPERATIONS
    # =========================================================================

    def copy_selected(self) -> None:
        if self.scene.selected_index >= 0: 
            self.clipboard = copy.deepcopy(self.scene.entities[self.scene.selected_index])
    
    def cut_selected(self) -> None:
        self.copy_selected()
        self.delete_selected()
        
    def _add_entity_recursive(self, ent: Entity) -> None:
        self.scene.add_entity(ent)
        for child in ent.children:
            self._add_entity_recursive(child)

    def paste_copied(self) -> None:
        if getattr(self, 'clipboard', None):
            light = self.clipboard.get_component(LightComponent)
            if light:
                ltype = light.type
                count = sum(1 for e in self.scene.entities if e.get_component(LightComponent) and e.get_component(LightComponent).type == ltype)
                limit = MAX_LIGHTS.get(ltype, 0)
                if count >= limit: 
                    raise SimulationError(f"Cannot paste {ltype} Light. Limit of {limit} reached.")
            
            new_ent = copy.deepcopy(self.clipboard)
            new_ent.name += " (Copy)"
            
            tf = new_ent.get_component(TransformComponent)
            if tf: 
                tf.position += glm.vec3(*PASTE_OFFSET) 
            
            self._add_entity_recursive(new_ent)
            return
            
        raise SimulationError("Cannot paste: The clipboard is currently empty.")
        
    def delete_selected(self) -> None:
        if self.scene.selected_index >= 0: 
            self.scene.remove_entity(self.scene.selected_index)

    def toggle_visibility_selected(self) -> None:
        if self.scene.selected_index >= 0:
            ent = self.scene.entities[self.scene.selected_index]
            light = ent.get_component(LightComponent)
            
            if light and light.type == "Directional": 
                return 
            
            mesh = ent.get_component(MeshRenderer)
            if mesh: 
                mesh.visible = not mesh.visible

    def toggle_all_lights(self, is_on: bool) -> None:
        for ent in self.scene.entities:
            light = ent.get_component(LightComponent)
            if light: 
                light.on = is_on

    def toggle_all_proxies(self, is_visible: bool) -> None:
        for ent in self.scene.entities:
            mesh = ent.get_component(MeshRenderer)
            if mesh and getattr(mesh, 'is_proxy', False): 
                mesh.visible = is_visible

    def sync_hierarchy_from_ui(self, hierarchy_mapping: Dict[int, Optional[int]]) -> None:
        for child_id, parent_id in hierarchy_mapping.items():
            if child_id >= len(self.scene.entities):
                continue
                
            child = self.scene.entities[child_id]
            current_parent = child.parent
            new_parent = self.scene.entities[parent_id] if parent_id is not None else None
            
            if current_parent != new_parent:
                if new_parent is not None and not new_parent.is_group:
                    continue

                if current_parent is not None:
                    current_parent.remove_child(child, keep_world=True)
                    
                if new_parent is not None:
                    new_parent.add_child(child, keep_world=True)

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    def load_texture_to_selected(self, map_attr: str, filepath: str) -> None:
        if self.scene.selected_index < 0: 
            raise SimulationError("Please select an entity in the scene first!")
            
        mesh = self.scene.entities[self.scene.selected_index].get_component(MeshRenderer)
        if mesh:
            tex_id = ResourceManager.load_texture(filepath)
            setattr(mesh.material, map_attr, tex_id)
            if not hasattr(mesh.material, 'tex_paths'):
                mesh.material.tex_paths = {}
            mesh.material.tex_paths[map_attr] = filepath

    def remove_texture_from_selected(self, map_attr: str) -> None:
        if self.scene.selected_index < 0: 
            return
            
        mesh = self.scene.entities[self.scene.selected_index].get_component(MeshRenderer)
        if mesh and getattr(mesh.material, map_attr, 0) != 0:
            setattr(mesh.material, map_attr, 0)
            if hasattr(mesh.material, 'tex_paths') and map_attr in mesh.material.tex_paths:
                del mesh.material.tex_paths[map_attr]

    def is_texture_in_use(self, path: str) -> bool:
        for ent in self.scene.entities:
            if self._check_texture_usage(ent, path):
                return True
        return False

    def _check_texture_usage(self, ent: Entity, path: str) -> bool:
        mesh = ent.get_component(MeshRenderer)
        if mesh and hasattr(mesh.material, 'tex_paths') and path in mesh.material.tex_paths.values():
            return True
            
        for child in ent.children:
            if self._check_texture_usage(child, path):
                return True
        return False

    # =========================================================================
    # SERIALIZATION & EXPORT
    # =========================================================================

    def get_scene_snapshot(self) -> str:
        entities_data = [self._serialize_entity(ent) for ent in self.scene.entities if ent.parent is None]
        return json.dumps(entities_data)

    def restore_snapshot(self, snapshot_str: str, current_aspect: float) -> None:
        if not snapshot_str: 
            return
            
        entities_data = json.loads(snapshot_str)
        self.scene.entities.clear()
        self.scene.selected_index = -1
        
        for ent_data in entities_data:
            ent = self._deserialize_entity(ent_data)
            self._add_entity_recursive(ent)

    def save_project(self, file_path: str, metadata: Dict[str, Any]) -> None:
        data = {
            "metadata": metadata,
            "semantic_classes": self.semantic_classes,
            "assets": {
                "models": list(ResourceManager.project_models),
                "textures": list(ResourceManager.project_textures)
            },
            "entities": [self._serialize_entity(ent) for ent in self.scene.entities if ent.parent is None]
        }

        with open(file_path, 'w', encoding='utf-8') as f: 
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_project(self, file_path: str, current_aspect: float) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)

        self.scene.entities.clear()
        self.scene.selected_index = -1
        ResourceManager.clear_project_assets()
        
        loaded_classes = data.get("semantic_classes", {
            0: {"name": "Car", "color": [1.0, 0.0, 0.0]}, 
            1: {"name": "Pedestrian", "color": [0.0, 1.0, 0.0]},
            2: {"name": "Traffic Sign", "color": [0.0, 0.0, 1.0]}, 
            3: {"name": "Misc", "color": [1.0, 1.0, 0.0]}
        })
        
        self.semantic_classes = {}
        for k, v in loaded_classes.items():
            if isinstance(v, str): 
                self.semantic_classes[int(k)] = {"name": v, "color": [0.8, 0.8, 0.8]}
            else:
                self.semantic_classes[int(k)] = v
        
        for p in data.get("assets", {}).get("models", []): 
            ResourceManager.add_project_model(p)
            
        for p in data.get("assets", {}).get("textures", []): 
            ResourceManager.add_project_texture(p)
        
        for ent_data in data.get("entities", []):
            ent = self._deserialize_entity(ent_data)
            self._add_entity_recursive(ent)
            
        return data.get("metadata", {})

    def _serialize_entity(self, ent: Entity) -> Dict[str, Any]:
        data = {"name": ent.name, "is_group": ent.is_group, "components": {}, "children": []}
        
        tf = ent.get_component(TransformComponent)
        if tf: data["components"]["transform"] = tf.to_dict()
            
        mesh = ent.get_component(MeshRenderer)
        if mesh: data["components"]["mesh"] = mesh.to_dict()
            
        light = ent.get_component(LightComponent)
        if light: data["components"]["light"] = light.to_dict()
            
        cam = ent.get_component(CameraComponent)
        if cam: data["components"]["camera"] = cam.to_dict()
            
        anim = ent.get_component(AnimationComponent)
        if anim and hasattr(anim, 'serialize'): 
            data["components"]["animation"] = anim.serialize()
            
        semantic = ent.get_component(SemanticComponent)
        if semantic and hasattr(semantic, 'serialize'): 
            data["components"]["semantic"] = semantic.serialize()
            
        for child in ent.children: 
            data["children"].append(self._serialize_entity(child))
            
        return data

    def _deserialize_entity(self, data: Dict[str, Any], parent: Optional[Entity] = None) -> Entity:
        ent = Entity(data["name"], is_group=data.get("is_group", False))
        comps = data.get("components", {})
        
        if "transform" in comps:
            tf = ent.add_component(TransformComponent())
            tf.from_dict(comps["transform"])
            
        if "mesh" in comps:
            renderer = ent.add_component(MeshRenderer())
            renderer.from_dict(comps["mesh"])
            
        if "light" in comps:
            l_comp = comps["light"]
            light = ent.add_component(LightComponent(light_type=l_comp.get("type", "Point")))
            light.from_dict(l_comp)

        if "camera" in comps:
            c_comp = comps["camera"]
            cam = ent.add_component(CameraComponent(mode=c_comp.get("mode", "Perspective")))
            cam.from_dict(c_comp)
            
        if "animation" in comps:
            anim = ent.add_component(AnimationComponent())
            if hasattr(anim, 'deserialize'): 
                anim.deserialize(comps["animation"])
                
        if "semantic" in comps:
            semantic = ent.add_component(SemanticComponent())
            if hasattr(semantic, 'deserialize'): 
                semantic.deserialize(comps["semantic"])
                
        for child_data in data.get("children", []):
            child_ent = self._deserialize_entity(child_data, ent) 
            ent.add_child(child_ent, keep_world=False)
            
        return ent

    def export_scene_obj(self, export_dir: str) -> None:
        obj_path = os.path.join(export_dir, "models.obj")
        mtl_path = os.path.join(export_dir, "materials.mtl")
        tex_folder_name = "textures" 
        tex_dir_full = os.path.join(export_dir, tex_folder_name)
        
        with open(obj_path, 'w', encoding='utf-8') as f_obj, open(mtl_path, 'w', encoding='utf-8') as f_mtl:
            f_obj.write("mtllib materials.mtl\n")
            export_state = {'v_offset': 1, 'mat_count': 0}
            
            try:
                for ent in self.scene.entities:
                    if ent.parent is None:
                        self._export_entity_to_obj(ent, f_obj, f_mtl, tex_dir_full, tex_folder_name, export_state)
            except Exception as e:
                raise ResourceError(f"Failed to export scene geometry to '{export_dir}'.\nReason: {e}")

    def _export_entity_to_obj(self, ent: Entity, f_obj: Any, f_mtl: Any, tex_dir_full: str, tex_folder_name: str, state: Dict[str, int]) -> None:
        mesh = ent.get_component(MeshRenderer)
        tf = ent.get_component(TransformComponent)
        
        if mesh and mesh.visible and not getattr(mesh, 'is_proxy', False) and mesh.geometry:
            mat_name = f"mat_{state['mat_count']}_{ent.name.replace(' ', '_')}"
            state['mat_count'] += 1
            mat = mesh.material
            
            f_mtl.write(f"newmtl {mat_name}\n")
            f_mtl.write(f"Ka {mat._ambient.x:.4f} {mat._ambient.y:.4f} {mat._ambient.z:.4f}\n")
            f_mtl.write(f"Kd {mat._diffuse.x:.4f} {mat._diffuse.y:.4f} {mat._diffuse.z:.4f}\n")
            f_mtl.write(f"Ks {mat._specular.x:.4f} {mat._specular.y:.4f} {mat._specular.z:.4f}\n")
            f_mtl.write(f"Ns {getattr(mat, 'shininess', 32.0):.4f}\n")
            
            tex_paths_dict = getattr(mat, 'tex_paths', {})
            for map_attr, t_path in tex_paths_dict.items():
                if t_path and os.path.exists(t_path):
                    tex_name = os.path.basename(t_path)
                    if not os.path.exists(tex_dir_full): 
                        os.makedirs(tex_dir_full)
                    dest_tex = os.path.join(tex_dir_full, tex_name)
                    try:
                        if not os.path.exists(dest_tex) or not os.path.samefile(t_path, dest_tex): 
                            shutil.copy2(t_path, dest_tex)
                        token = MTL_TOKENS.get(map_attr, "map_Kd")
                        f_mtl.write(f"{token} {tex_folder_name}/{tex_name}\n")
                    except Exception as e:
                        raise ResourceError(f"Texture packaging failed during export: {e}")
            f_mtl.write("\n")
            f_obj.write(f"o {ent.name.replace(' ', '_')}\n")
            f_obj.write(f"usemtl {mat_name}\n")
            
            geom = mesh.geometry
            if not hasattr(geom, 'vertices') or not hasattr(geom, 'vertex_size'): 
                return
                
            v_size = geom.vertex_size
            verts = np.array(geom.vertices, dtype=np.float32).reshape(-1, v_size)
            num_v = len(verts)
            global_mat = tf.get_matrix() if tf else glm.mat4(1.0)
            norm_mat = glm.transpose(glm.inverse(glm.mat3(global_mat)))
            
            for i in range(num_v):
                pos = glm.vec3(verts[i][0], verts[i][1], verts[i][2])
                g_pos = glm.vec3(global_mat * glm.vec4(pos, 1.0))
                f_obj.write(f"v {g_pos.x:.6f} {g_pos.y:.6f} {g_pos.z:.6f}\n")
                
            for i in range(num_v): 
                f_obj.write(f"vt {verts[i][6]:.6f} {verts[i][7]:.6f}\n")
                
            for i in range(num_v):
                norm = glm.vec3(verts[i][3], verts[i][4], verts[i][5])
                g_norm = glm.normalize(norm_mat * norm) if glm.length(norm) > 0 else glm.vec3(0, 1, 0)
                f_obj.write(f"vn {g_norm.x:.6f} {g_norm.y:.6f} {g_norm.z:.6f}\n")
                
            if getattr(geom, 'indices', None) is not None and len(geom.indices) > 0:
                idx = geom.indices
                v_off = state['v_offset']
                for i in range(0, len(idx), 3):
                    i1, i2, i3 = idx[i], idx[i+1], idx[i+2]
                    f_obj.write(f"f {i1+v_off}/{i1+v_off}/{i1+v_off} {i2+v_off}/{i2+v_off}/{i2+v_off} {i3+v_off}/{i3+v_off}/{i3+v_off}\n")
            else:
                v_off = state['v_offset']
                for i in range(num_v): 
                    f_obj.write(f"p {i+v_off}\n")
                    
            state['v_offset'] += num_v
            
        for child in ent.children:
            self._export_entity_to_obj(child, f_obj, f_mtl, tex_dir_full, tex_folder_name, state)