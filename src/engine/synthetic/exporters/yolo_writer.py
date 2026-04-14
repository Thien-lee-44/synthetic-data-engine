import os
from typing import List, Tuple

class YOLOWriter:
    """
    Serializes 2D Bounding Box data into the standard YOLO format:
    <class_id> <x_center> <y_center> <width> <height> (All values normalized 0.0 to 1.0)
    """

    @staticmethod
    def export(filepath: str, bboxes: List[Tuple[int, float, float, float, float]], img_w: int, img_h: int) -> None:
        """
        :param bboxes: List of tuples (class_id, x_min, y_min, x_max, y_max)
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for cls_id, xmin, ymin, xmax, ymax in bboxes:
                # Calculate absolute center and dimensions
                abs_w = xmax - xmin
                abs_h = ymax - ymin
                abs_cx = xmin + (abs_w / 2.0)
                abs_cy = ymin + (abs_h / 2.0)

                # Normalize to [0.0, 1.0] domain
                norm_cx = abs_cx / img_w
                norm_cy = abs_cy / img_h
                norm_w = abs_w / img_w
                norm_h = abs_h / img_h

                # Ensure values strictly remain within bounds to prevent AI training crashes
                norm_cx = max(0.0, min(1.0, norm_cx))
                norm_cy = max(0.0, min(1.0, norm_cy))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))

                f.write(f"{cls_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")