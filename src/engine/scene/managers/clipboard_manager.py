import copy
import glm
from typing import Any, Optional

from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, LightComponent
from src.app.exceptions import SimulationError
from src.app.config import MAX_LIGHTS, PASTE_OFFSET


class ClipboardManager:
    """
    Handles copy, cut, paste, and delete operations for entities.
    """

    def __init__(self, scene: Any, scene_mgr: Any) -> None:
        self.scene = scene
        self.scene_mgr = scene_mgr
        self.clipboard: Optional[Entity] = None

    def has_clipboard(self) -> bool:
        return self.clipboard is not None

    def copy_selected(self) -> None:
        if self.scene.selected_index >= 0:
            self.clipboard = copy.deepcopy(self.scene.entities[self.scene.selected_index])

    def cut_selected(self) -> None:
        self.copy_selected()
        self.delete_selected()

    def paste_copied(self) -> None:
        if not self.clipboard:
            raise SimulationError("Cannot paste: The clipboard is currently empty.")

        light = self.clipboard.get_component(LightComponent)
        if light:
            ltype = light.type
            count = sum(
                1
                for e in self.scene.entities
                if e.get_component(LightComponent) and e.get_component(LightComponent).type == ltype
            )
            limit = MAX_LIGHTS.get(ltype, 0)
            if count >= limit:
                raise SimulationError(f"Cannot paste {ltype} Light. Limit of {limit} reached.")

        new_ent = copy.deepcopy(self.clipboard)
        new_ent.name += " (Copy)"

        tf = new_ent.get_component(TransformComponent)
        if tf:
            tf.position += glm.vec3(*PASTE_OFFSET)

        self.scene_mgr._add_entity_recursive(new_ent)

    def delete_selected(self) -> None:
        if self.scene.selected_index >= 0:
            self.scene.remove_entity(self.scene.selected_index)

