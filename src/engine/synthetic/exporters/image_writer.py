import os
import numpy as np
from PIL import Image

class ImageWriter:
    """
    Handles the serialization of raw OpenGL pixel buffers to standard image formats.
    Automatically rectifies OpenGL's bottom-left origin to the standard top-left origin.
    """

    @staticmethod
    def save_rgb(filepath: str, pixel_data: bytes, width: int, height: int) -> None:
        """Saves a standard 3-channel RGB image."""
        arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(arr, 'RGB')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filepath, format="JPEG", quality=95)

    @staticmethod
    def save_mask(filepath: str, pixel_data: bytes, width: int, height: int) -> None:
        """Saves a 3-channel Instance/Semantic Mask image (Lossless PNG)."""
        arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(arr, 'RGB')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filepath, format="PNG")

    @staticmethod
    def save_depth(filepath: str, depth_data: np.ndarray, width: int, height: int) -> None:
        """
        Saves a 1-channel Depth Map. 
        Normalizes the non-linear OpenGL depth buffer for clear visual perception.
        """
        # 1. Ignore background (1.0) to find the actual min/max of the scene's geometry
        valid_depths = depth_data[depth_data < 1.0]
        
        if len(valid_depths) > 0:
            d_min = valid_depths.min()
            d_max = valid_depths.max()
            if d_max > d_min:
                # Normalize the valid depths to span the full 0.0 to 1.0 range
                depth_data = np.where(depth_data < 1.0, (depth_data - d_min) / (d_max - d_min), 1.0)
        
        # 2. Invert depth so closer objects appear brighter (Standard computer vision practice)
        depth_data = 1.0 - depth_data 
        
        # 3. Convert to 8-bit Grayscale
        depth_8bit = (depth_data * 255).astype(np.uint8).reshape((height, width))
        img = Image.fromarray(depth_8bit, 'L')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filepath, format="PNG")