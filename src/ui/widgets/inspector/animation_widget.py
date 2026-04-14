from typing import Any, Dict
from PySide6.QtWidgets import QFormLayout, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QCheckBox, QWidget
from src.ui.widgets.inspector.base_widget import BaseComponentWidget

class AnimationWidget(BaseComponentWidget):
    """
    Inspector UI component for granular keyframe management.
    Displays individual keyframes and allows localized deletion and playback looping control.
    """
    def __init__(self, controller: Any) -> None:
        super().__init__("Animation & Keyframes", controller)
        
        form = QFormLayout()
        form.setContentsMargins(0, 5, 0, 5)

        self.chk_loop = QCheckBox("Loop Playback")
        self.chk_loop.toggled.connect(self._on_loop_toggled)
        form.addRow("", self.chk_loop)
        
        self.lbl_duration = QLabel("Duration: 0.0s")
        self.lbl_duration.setStyleSheet("color: #888; font-style: italic;")
        form.addRow("Total Time:", self.lbl_duration)

        self.layout.addLayout(form)
        
        self.kf_container = QVBoxLayout()
        self.kf_container.setSpacing(2)
        self.kf_container.setContentsMargins(0, 5, 0, 0)
        self.layout.addLayout(self.kf_container)

    def update_data(self, data: Dict[str, Any]) -> None:
        self.chk_loop.blockSignals(True)
        self.chk_loop.setChecked(data.get("loop", True))
        self.chk_loop.blockSignals(False)
        
        keyframes = data.get("keyframes", [])
        duration = keyframes[-1].get("time", 0.0) if keyframes else 0.0
        self.lbl_duration.setText(f"{duration:.2f} s")
        
        while self.kf_container.count():
            item = self.kf_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        for i, kf in enumerate(keyframes):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            
            lbl = QLabel(f"Keyframe [{i}]  ->  t = {kf.get('time', 0.0):.2f}s")
            lbl.setStyleSheet("font-size: 11px;")
            
            btn_del = QPushButton("x")
            btn_del.setFixedWidth(20)
            btn_del.setFixedHeight(20)
            btn_del.setStyleSheet("color: #ff4444; font-weight: bold; border: 1px solid #ff4444; border-radius: 2px;")
            btn_del.setCursor(btn_del.cursor())
            
            btn_del.clicked.connect(lambda checked=False, idx=i: self._remove_kf(idx))
            
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(btn_del)
            
            w = QWidget()
            w.setLayout(row)
            self.kf_container.addWidget(w)
            
    def _on_loop_toggled(self, checked: bool) -> None:
        if self._controller:
            self._controller.request_undo_snapshot()
            self._controller.set_property("Animation", "loop", checked)
            
    def _remove_kf(self, idx: int) -> None:
        if self._controller:
            self._controller.remove_keyframe(idx)