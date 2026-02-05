import sys
import os
import torch
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QWidget
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation
)
import matplotlib


def visualize_segmentation(image, predicted_instance_map):
    """
    Overlays instance segmentation masks on an image.
    """
    image = image.convert("RGBA")

    # Get unique instance IDs
    # excluding a potential background class if necessary.
    instance_ids = torch.unique(predicted_instance_map)
    instance_ids = instance_ids[instance_ids != 0]  # Assuming 0 is background

    n_masks = len(instance_ids)
    if n_masks == 0:
        return image.convert("RGB")

    # Generate colors for each instance
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    # Overlay each mask
    for i, instance_id in enumerate(instance_ids):
        # Create a binary mask for the current instance
        mask_tensor = (predicted_instance_map == instance_id)
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255

        mask_img = Image.fromarray(mask_np, 'L')

        # Create a colored overlay for the mask
        overlay = Image.new("RGBA", image.size, colors[i] + (0,))

        # Apply a 50% transparent alpha channel to the overlay
        alpha = mask_img.point(lambda p: p * 0.5)
        overlay.putalpha(alpha)

        # Composite the overlay onto the image
        image = Image.alpha_composite(image, overlay)

    return image.convert("RGB")


class Mask2FormerApp(QMainWindow):
    """Mask2Former 모델을 위한 PyQt6 기반 GUI 애플리케이션"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Mask2Former Instance Segmentation")
        self.setGeometry(100, 100, 1600, 800)

        # --- 모델 로딩 ---
        self.statusBar().showMessage("Mask2Former 모델을 로딩 중입니다...")
        QApplication.processEvents()  # UI 업데이트 강제

        model_name = "facebook/mask2former-swin-large-coco-instance"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name)

        self.statusBar().showMessage("모델 로딩 완료.", 5000)

        # --- 상태 변수 ---
        self.current_original_pixmap = None
        self.current_segmented_pixmap = None

        # --- UI 초기화 ---
        self.init_ui()

        # --- 이미지 목록 채우기 ---
        self.populate_image_list()

    def init_ui(self):
        # 메인 스플리터 (좌/우 분할)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- 왼쪽 패널 (이미지 목록) ---
        self.image_list_widget = QListWidget()
        self.image_list_widget.currentItemChanged.connect(
            self.on_image_selected)

        # --- 오른쪽 패널 (이미지 표시) ---
        right_pane = QWidget()
        right_layout = QHBoxLayout(right_pane)

        self.original_image_label = QLabel("원본 이미지")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setMinimumSize(400, 400)

        self.segmented_image_label = QLabel("분할 결과")
        self.segmented_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.segmented_image_label.setMinimumSize(400, 400)

        right_layout.addWidget(self.original_image_label)
        right_layout.addWidget(self.segmented_image_label)

        # 스플리터에 위젯 추가
        splitter.addWidget(self.image_list_widget)
        splitter.addWidget(right_pane)
        splitter.setSizes([300, 1300])  # 초기 크기 설정

        self.setCentralWidget(splitter)

    def populate_image_list(self, directory="data"):
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.image_list_widget.addItem("'data' 폴더가 없습니다. 이미지를 추가해주세요.")
            return

        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_files = [f for f in os.listdir(
            directory) if f.lower().endswith(supported_formats)]

        if not image_files:
            self.image_list_widget.addItem("'data' 폴더에 이미지가 없습니다.")
            return

        for filename in image_files:
            item = QListWidgetItem(filename)
            item.setData(Qt.ItemDataRole.UserRole,
                         os.path.join(directory, filename))
            self.image_list_widget.addItem(item)

    def on_image_selected(self, current_item, _):
        if not current_item or not current_item.data(Qt.ItemDataRole.UserRole):
            return

        image_path = current_item.data(Qt.ItemDataRole.UserRole)
        self.statusBar().showMessage(
            f"이미지 처리 중: {os.path.basename(image_path)}...")
        QApplication.processEvents()

        original_pil_image = Image.open(image_path).convert("RGB")

        # --- 원본 이미지 표시 ---
        self.current_original_pixmap = self.pil_to_qpixmap(original_pil_image)
        self.update_image_display()

        # --- Mask2Former 추론 실행 ---
        inputs = self.processor(images=original_pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # --- 후처리 및 시각화 ---
        result = self.processor.post_process_instance_segmentation(
            outputs, target_sizes=[
                original_pil_image.size[::-1]], threshold=0.2
        )[0]

        predicted_instance_map = result["segmentation"]

        segmented_image = visualize_segmentation(
            original_pil_image.copy(), predicted_instance_map)

        self.current_segmented_pixmap = self.pil_to_qpixmap(segmented_image)
        self.update_image_display()

        self.statusBar().showMessage("이미지 처리 완료.", 5000)

    def update_image_display(self):
        if self.current_original_pixmap:
            self.original_image_label.setPixmap(
                self.current_original_pixmap.scaled(
                    self.original_image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
        if self.current_segmented_pixmap:
            self.segmented_image_label.setPixmap(
                self.current_segmented_pixmap.scaled(
                    self.segmented_image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))

    def pil_to_qpixmap(self, pil_img):
        rgb_image = pil_img.convert("RGB")
        h, w, ch = rgb_image.height, rgb_image.width, 3
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.tobytes(), w, h,
                         bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_display()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = Mask2FormerApp()
    main_win.show()
    sys.exit(app.exec())
