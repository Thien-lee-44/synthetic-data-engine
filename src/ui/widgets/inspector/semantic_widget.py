from typing import Any, Dict
from PySide6.QtWidgets import QFormLayout, QLabel, QComboBox, QPushButton, QHBoxLayout, QInputDialog, QColorDialog
from PySide6.QtGui import QColor
from src.ui.widgets.inspector.base_widget import BaseComponentWidget

class SemanticWidget(BaseComponentWidget):
    """
    Inspector UI component dedicated to AI data labeling (Class ID) and Tracking ID.
    Now supports custom colors for each semantic class.
    """
    def __init__(self, controller: Any) -> None:
        super().__init__("Semantic Labeling", controller)
        
        form = QFormLayout()
        form.setContentsMargins(0, 5, 0, 5)

        self.lbl_track_id = QLabel("-1")
        self.lbl_track_id.setStyleSheet("color: #4CAF50; font-weight: bold;")
        form.addRow("Track ID (Auto):", self.lbl_track_id)

        self.cmb_class = QComboBox()
        self.cmb_class.currentIndexChanged.connect(self.apply_changes)
        
        self.btn_color = QPushButton()
        self.btn_color.setFixedWidth(28)
        self.btn_color.setToolTip("Change Class Color")
        self.btn_color.clicked.connect(self.change_class_color)

        self.btn_add = QPushButton("+")
        self.btn_add.setFixedWidth(28)
        self.btn_add.setToolTip("Add new Semantic Class")
        self.btn_add.clicked.connect(self.add_new_class)
        
        class_layout = QHBoxLayout()
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.addWidget(self.cmb_class)
        class_layout.addWidget(self.btn_color)
        class_layout.addWidget(self.btn_add)

        form.addRow("Class ID:", class_layout)
        self.layout.addLayout(form)

    def update_data(self, data: Dict[str, Any]) -> None:
        self.lbl_track_id.setText(str(data.get("track_id", -1)))
        
        self.cmb_class.blockSignals(True)
        self.cmb_class.clear()
        
        if self._controller:
            classes = self._controller.get_semantic_classes()
            for c_id, c_info in classes.items():
                name = c_info.get("name", "Unknown") if isinstance(c_info, dict) else c_info
                self.cmb_class.addItem(f"{c_id}: {name}", c_id)

        target_id = data.get("class_id", 0)
        idx = self.cmb_class.findData(target_id)
        if idx >= 0:
            self.cmb_class.setCurrentIndex(idx)
            
            c_info = classes.get(target_id, {}) if self._controller else {}
            color = c_info.get("color", [1.0, 1.0, 1.0]) if isinstance(c_info, dict) else [1.0, 1.0, 1.0]
            r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            self.btn_color.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #555; border-radius: 3px;")
            
        self.cmb_class.blockSignals(False)

    def apply_changes(self) -> None:
        if self._controller:
            class_id = self.cmb_class.currentData()
            if class_id is not None:
                self._controller.request_undo_snapshot()
                self._controller.set_property("Semantic", "class_id", class_id)
                
                classes = self._controller.get_semantic_classes()
                c_info = classes.get(class_id, {})
                color = c_info.get("color", [1.0, 1.0, 1.0]) if isinstance(c_info, dict) else [1.0, 1.0, 1.0]
                r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                self.btn_color.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #555; border-radius: 3px;")

    def add_new_class(self) -> None:
        if not self._controller: 
            return
            
        name, ok = QInputDialog.getText(self, "New Semantic Class", "Enter class name (e.g. Tree, Road):")
        if ok and name.strip():
            new_id = self._controller.add_semantic_class(name.strip())
            self.cmb_class.blockSignals(True)
            self.cmb_class.addItem(f"{new_id}: {name.strip()}", new_id)
            idx = self.cmb_class.findData(new_id)
            self.cmb_class.setCurrentIndex(idx)
            self.cmb_class.blockSignals(False)
            self.apply_changes()

    def change_class_color(self) -> None:
        if not self._controller: 
            return
            
        class_id = self.cmb_class.currentData()
        if class_id is None: 
            return
        
        classes = self._controller.get_semantic_classes()
        c_info = classes.get(class_id, {})
        curr_color = c_info.get("color", [1.0, 1.0, 1.0]) if isinstance(c_info, dict) else [1.0, 1.0, 1.0]
        
        init_color = QColor(int(curr_color[0] * 255), int(curr_color[1] * 255), int(curr_color[2] * 255))
        color = QColorDialog.getColor(init_color, self, "Select Class Color")
        
        if color.isValid():
            rgb = [color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0]
            self._controller.update_semantic_class_color(class_id, rgb)
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #555; border-radius: 3px;")