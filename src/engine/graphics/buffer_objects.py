import ctypes
import numpy as np
import glm
from OpenGL.GL import *
from typing import Optional, List, Union

from src.app.config import DEFAULT_POINT_SIZE

class BufferObject:
    """
    Encapsulates OpenGL Vertex Array Objects (VAO), Vertex Buffer Objects (VBO), 
    and Element Buffer Objects (EBO). 
    Architectural Note: This class strictly manages raw geometry data transmission 
    to the GPU memory and computes spatial bounds for AI annotation.
    """
    
    def __init__(self, vertices: Union[List[float], np.ndarray], indices: Optional[Union[List[int], np.ndarray]] = None, vertex_size: int = 8, render_mode: int = GL_TRIANGLES) -> None:
        self.vertex_size = vertex_size 
        self.render_mode = render_mode
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.has_vertex_color = (vertex_size >= 11)
        
        # [NEW] Compute Axis-Aligned Bounding Box (AABB) bounds directly from vertices
        self._compute_aabb()
        
        if indices is not None and len(indices) > 0:
            self.indices = np.ascontiguousarray(indices, dtype=np.uint32)
        else:
            self.indices = None
            
        self._setup_opengl_buffers()

    def _compute_aabb(self) -> None:
        """
        Dynamically calculates the Axis-Aligned Bounding Box limits of the raw geometry.
        Crucial for precise 2D YOLO bounding box projection in the Synthetic Generator.
        """
        if len(self.vertices) == 0 or self.vertex_size < 3:
            self.aabb_min = glm.vec3(-0.5, -0.5, -0.5)
            self.aabb_max = glm.vec3(0.5, 0.5, 0.5)
            return
            
        # Reshape the flat 1D vertex array into a 2D matrix (Rows: Vertices, Cols: Attributes)
        vertices_reshaped = self.vertices.reshape(-1, self.vertex_size)
        
        # Extract only the XYZ spatial positional data (first 3 columns)
        positions = vertices_reshaped[:, 0:3]
        
        # Calculate the absolute spatial extents using vectorized NumPy operations
        min_vals = np.min(positions, axis=0)
        max_vals = np.max(positions, axis=0)
        
        self.aabb_min = glm.vec3(min_vals[0], min_vals[1], min_vals[2])
        self.aabb_max = glm.vec3(max_vals[0], max_vals[1], max_vals[2])

    def _setup_opengl_buffers(self) -> None:
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(
            GL_ARRAY_BUFFER, 
            self.vertices.nbytes, 
            self.vertices.ctypes.data_as(ctypes.c_void_p), 
            GL_STATIC_DRAW
        )
        
        if self.indices is not None:
            self.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER, 
                self.indices.nbytes, 
                self.indices.ctypes.data_as(ctypes.c_void_p), 
                GL_STATIC_DRAW
            )
            
        stride = self.vertex_size * 4 
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
        
        if self.has_vertex_color:
            glEnableVertexAttribArray(3)
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8 * 4))
        else:
            glDisableVertexAttribArray(3)
            glVertexAttrib3f(3, -1.0, -1.0, -1.0) 
            
        glBindVertexArray(0)

    def draw(self) -> None:
        glBindVertexArray(self.vao)
        if self.indices is not None:
            glDrawElements(self.render_mode, len(self.indices), GL_UNSIGNED_INT, None)
        else:
            if self.render_mode == GL_POINTS:
                glPointSize(DEFAULT_POINT_SIZE) 
                
            vertex_count = len(self.vertices) // self.vertex_size
            glDrawArrays(self.render_mode, 0, vertex_count)
        glBindVertexArray(0)
        
    def delete_buffers(self) -> None:
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        if self.indices is not None:
            glDeleteBuffers(1, [self.ebo])