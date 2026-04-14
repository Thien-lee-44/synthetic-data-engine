import glm
from typing import Tuple, Optional, Any

class LabelUtils:
    """
    Provides mathematical utilities to project 3D spatial data into 2D screen-space bounds.
    Critical for automating Ground Truth annotation (YOLO/COCO formats).
    """

    @staticmethod
    def get_2d_bounding_box(entity_tf: Any, mesh_geom: Any, view_mat: glm.mat4, proj_mat: glm.mat4, screen_w: int, screen_h: int) -> Optional[Tuple[float, float, float, float]]:
        """
        Projects an entity's 3D Axis-Aligned Bounding Box (AABB) into 2D pixel coordinates.
        
        :return: Tuple containing (X_min, Y_min, X_max, Y_max) or None if fully off-screen.
        """
        if not mesh_geom: 
            return None

        # Attempt to retrieve pre-calculated local AABB bounds from the geometry buffer.
        # Fallback to a normalized unit cube if bounds are missing.
        min_p = getattr(mesh_geom, 'aabb_min', glm.vec3(-0.5, -0.5, -0.5))
        max_p = getattr(mesh_geom, 'aabb_max', glm.vec3(0.5, 0.5, 0.5))
        
        # Define the 8 corner vertices of the local 3D bounding box
        corners = [
            glm.vec4(min_p.x, min_p.y, min_p.z, 1.0),
            glm.vec4(max_p.x, min_p.y, min_p.z, 1.0),
            glm.vec4(min_p.x, max_p.y, min_p.z, 1.0),
            glm.vec4(max_p.x, max_p.y, min_p.z, 1.0),
            glm.vec4(min_p.x, min_p.y, max_p.z, 1.0),
            glm.vec4(max_p.x, min_p.y, max_p.z, 1.0),
            glm.vec4(min_p.x, max_p.y, max_p.z, 1.0),
            glm.vec4(max_p.x, max_p.y, max_p.z, 1.0),
        ]

        # Model-View-Projection Matrix
        mvp = proj_mat * view_mat * entity_tf.get_matrix()
        
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        points_behind_camera = 0

        for corner in corners:
            # Transform to Clip Space
            clip = mvp * corner
            
            # W-clipping: Object intersects or is behind the near plane
            if clip.w <= 0.0:
                points_behind_camera += 1
                continue
                
            # Perspective Division to Normalized Device Coordinates (NDC)
            ndc = clip / clip.w
            
            # Remap NDC [-1, 1] to Screen Space Pixels [0, W] and [0, H]
            # Y-axis is inverted to match standard image coordinate systems (top-left origin)
            sx = (ndc.x + 1.0) * 0.5 * screen_w
            sy = (1.0 - ndc.y) * 0.5 * screen_h 

            min_x = min(min_x, sx)
            min_y = min(min_y, sy)
            max_x = max(max_x, sx)
            max_y = max(max_y, sy)

        # If the entire bounding box is behind the camera, cull it completely
        if points_behind_camera == 8:
            return None

        # Clamp extremes to physical image boundaries to prevent out-of-bounds annotations
        min_x = max(0.0, min_x)
        min_y = max(0.0, min_y)
        max_x = min(float(screen_w), max_x)
        max_y = min(float(screen_h), max_y)

        # Validate that the bounding box has physical dimensions (not squashed or fully culled)
        if min_x >= max_x or min_y >= max_y:
            return None

        return (min_x, min_y, max_x, max_y)