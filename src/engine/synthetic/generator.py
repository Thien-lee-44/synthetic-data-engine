import os
import glm
import numpy as np
from typing import Any, Callable, Optional

from src.engine.synthetic.exporters.image_writer import ImageWriter
from src.engine.synthetic.exporters.yolo_writer import YOLOWriter
from src.engine.synthetic.label_utils import LabelUtils

from src.engine.scene.components import TransformComponent, MeshRenderer
from src.engine.scene.components.semantic_cmp import SemanticComponent

class SyntheticDataGenerator:
    """
    The orchestrator for the Synthetic Dataset Generation pipeline.
    Automates the temporal advancement of the scene, frame capture, and ground truth extraction.
    Operates directly on the core subsystems (Scene, Renderer, Animator) bypassing the Engine UI Facade.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine 
        self.scene = engine.scene
        self.renderer = engine.renderer
        self.animator = getattr(engine, 'animator', None)
        
        self.is_running = False
        self.output_dir = ""

    def setup_directories(self, base_path: str) -> None:
        """Prepares the target dataset folder hierarchy and writes the dynamic YOLO dataset.yaml"""
        self.output_dir = base_path
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "depth"), exist_ok=True)

        yaml_path = os.path.join(self.output_dir, "dataset.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("path: ./  # Adjust to absolute path if needed\n")
            f.write("train: images\n")
            f.write("val: images\n\n")
            f.write("names:\n")
            
            classes = getattr(self.engine, 'semantic_classes', {0: {"name": "Car"}})
            for c_id, c_info in classes.items():
                name = c_info.get("name", "Unknown") if isinstance(c_info, dict) else c_info
                f.write(f"  {c_id}: {name}\n")

    def generate_batch(self, num_frames: int, dt: float, res_w: int = 1024, res_h: int = 1024, progress_cb: Optional[Callable[[int], None]] = None) -> None:
        """Executes the synchronous batch generation loop."""
        if not self.scene or not self.renderer:
            raise RuntimeError("Cannot generate dataset: Core subsystems (Scene/Renderer) are missing.")

        self.is_running = True
        
        active_camera = None
        for tf, cam, ent in self.scene.cached_cameras:
            if getattr(cam, 'is_active', False):
                active_camera = cam
                break
        
        if not active_camera:
            self.is_running = False
            raise RuntimeError("Cannot generate dataset: No active camera found in the scene.")

        old_aspect = getattr(active_camera, 'aspect_ratio', getattr(active_camera, 'aspect', None))
        if hasattr(active_camera, 'aspect_ratio'):
            active_camera.aspect_ratio = res_w / res_h
        elif hasattr(active_camera, 'aspect'):
            active_camera.aspect = res_w / res_h
            
        if hasattr(active_camera, 'update_projection_matrix'):
            active_camera.update_projection_matrix()

        try:
            current_sim_time = 0.0 

            for frame_idx in range(num_frames):
                if not self.is_running:
                    break

                try:
                    from src.app import ctx
                    if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
                        ctx.main_window.gl_widget.makeCurrent()
                except ImportError:
                    pass

                if self.animator:
                    self.animator.evaluate(current_sim_time, dt) 

                view_mat = active_camera.get_view_matrix()
                proj_mat = active_camera.get_projection_matrix()
                
                merged_bboxes = {}
                
                for ent in self.scene.entities:
                    semantic = ent.get_component(SemanticComponent)
                    tf = ent.get_component(TransformComponent)
                    mesh = ent.get_component(MeshRenderer)
                    
                    if semantic and tf and mesh and mesh.visible:
                        bbox = LabelUtils.get_2d_bounding_box(tf, mesh.geometry, view_mat, proj_mat, res_w, res_h)
                        if bbox:
                            xmin, ymin, xmax, ymax = bbox
                            t_id = semantic.track_id
                            c_id = semantic.class_id
                            
                            if t_id in merged_bboxes:
                                curr = merged_bboxes[t_id]
                                merged_bboxes[t_id] = [
                                    c_id,
                                    min(curr[1], xmin),
                                    min(curr[2], ymin),
                                    max(curr[3], xmax),
                                    max(curr[4], ymax)
                                ]
                            else:
                                merged_bboxes[t_id] = [c_id, xmin, ymin, xmax, ymax]

                bboxes = list(merged_bboxes.values())

                label_path = os.path.join(self.output_dir, "labels", f"frame_{frame_idx:05d}.txt")
                YOLOWriter.export(label_path, bboxes, res_w, res_h)

                rgb_pixels = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="RGB")
                mask_pixels = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="MASK")
                depth_pixels = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="DEPTH")
                
                img_path = os.path.join(self.output_dir, "images", f"frame_{frame_idx:05d}.jpg")
                mask_path = os.path.join(self.output_dir, "masks", f"frame_{frame_idx:05d}.png")
                depth_path = os.path.join(self.output_dir, "depth", f"frame_{frame_idx:05d}.png")
                
                if rgb_pixels is not None and len(rgb_pixels) > 0:
                    ImageWriter.save_rgb(img_path, rgb_pixels, res_w, res_h)
                    
                if mask_pixels is not None and len(mask_pixels) > 0:
                    ImageWriter.save_mask(mask_path, mask_pixels, res_w, res_h)
                
                if depth_pixels is not None and len(depth_pixels) > 0:
                    depth_arr = np.frombuffer(depth_pixels, dtype=np.float32)
                    ImageWriter.save_depth(depth_path, depth_arr, res_w, res_h)
                
                current_sim_time += dt 

                if progress_cb:
                    progress_cb(frame_idx + 1)

                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()

        finally:
            self.is_running = False
            if old_aspect is not None:
                if hasattr(active_camera, 'aspect_ratio'):
                    active_camera.aspect_ratio = old_aspect
                elif hasattr(active_camera, 'aspect'):
                    active_camera.aspect = old_aspect
                if hasattr(active_camera, 'update_projection_matrix'):
                    active_camera.update_projection_matrix()