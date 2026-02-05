import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

from PIL import Image
from sam3.model_builder import build_sam3_image_model  # type: ignore
from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
import numpy as np
import matplotlib


def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks.shape[0]
    if n_masks == 0:
        return image.convert("RGB")

    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for i, (mask, color) in enumerate(zip(masks, colors)):
        mask_2d = np.squeeze(mask)

        mask_img = Image.fromarray(mask_2d)
        overlay = Image.new("RGBA", image.size, color + (0,))

        # 마스크 영역에 50% 불투명도의 알파 채널 생성
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)

        image = Image.alpha_composite(image, overlay)

    return image.convert("RGB")


class SegmentationApp(QMainWindow):
    """SAM3 모델을 위한 PyQt6 기반 GUI 애플리케이션"""

    def __init__(self):
        super().__init__()

        print("Initializing window object...")

        self.setWindowTitle("SAM3 Segmentation GUI")
        self.setGeometry(100, 100, 1200, 800)

        # --- 모델 및 프로세서 로딩 ---
        self.statusBar().showMessage("SAM3 모델을 로딩 중입니다...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        self.statusBar().showMessage("모델 로딩 완료.", 5000)

        # --- 상태 변수 ---
        self.original_pil_image = None
        self.inference_state = None
        self.current_pixmap = None  # 원본 해상도 QPixmap 저장용

        # --- UI 초기화 ---
        self.init_ui()

        # --- 이미지 목록 채우기 ---
        self.populate_image_list()

    def init_ui(self):

        print("Initializing gui...")

        # 메인 스플리터 (좌/우 분할)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- 왼쪽 패널 ---
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)

        # 이미지 목록 위젯
        self.image_list_widget = QListWidget()
        self.image_list_widget.currentItemChanged.connect(
            self.on_image_selected)
        left_layout.addWidget(self.image_list_widget)

        # 하단 컨트롤 (프롬프트 입력 및 버튼)
        bottom_controls_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_controls_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("텍스트 프롬프트를 입력하세요...")
        bottom_layout.addWidget(self.prompt_input)

        self.segment_button = QPushButton("Segment")
        self.segment_button.clicked.connect(self.run_segmentation)
        bottom_layout.addWidget(self.segment_button)

        left_layout.addWidget(bottom_controls_widget)

        # --- 오른쪽 패널 (이미지 표시) ---
        self.image_label = QLabel("왼쪽 목록에서 이미지를 선택하세요.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)

        # 스플리터에 패널 추가
        splitter.addWidget(left_pane)
        splitter.addWidget(self.image_label)
        splitter.setSizes([300, 900])  # 초기 크기 설정

        self.setCentralWidget(splitter)

    def populate_image_list(self, directory="data"):

        print("Reading image list...")

        """'data' 디렉토리에서 이미지를 찾아 목록에 추가합니다."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.image_list_widget.addItem(
                "'data' 폴더가 없습니다. 생성 후 이미지를 추가해주세요.")
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
        """이미지 목록에서 항목 선택 시 호출됩니다."""
        if not current_item or not current_item.data(Qt.ItemDataRole.UserRole):
            return

        image_path = current_item.data(Qt.ItemDataRole.UserRole)
        self.statusBar().showMessage(
            f"이미지 로딩 중: {os.path.basename(image_path)}...")

        self.original_pil_image = Image.open(image_path).convert("RGB")
        self.inference_state = self.processor.set_image(
            self.original_pil_image)

        self.current_pixmap = self.pil_to_qpixmap(self.original_pil_image)
        self.update_image_display()
        self.statusBar().showMessage("이미지 로딩 완료. Segmentation 준비 완료.", 5000)

    def run_segmentation(self):
        """'Segment' 버튼 클릭 시 Segmentation을 실행합니다."""
        if not self.inference_state or not self.original_pil_image:
            self.statusBar().showMessage("먼저 이미지를 선택해주세요.", 3000)
            return

        prompt = self.prompt_input.text().strip()
        if not prompt:
            self.statusBar().showMessage("텍스트 프롬프트를 입력해주세요.", 3000)
            return

        self.statusBar().showMessage(f"'{prompt}' 프롬프트로 Segmentation 실행 중...")
        QApplication.processEvents()  # UI가 메시지를 표시하도록 강제 업데이트

        output = self.processor.set_text_prompt(
            state=self.inference_state, prompt=prompt)
        masks = output["masks"]

        if masks.shape[0] == 0:
            self.statusBar().showMessage(
                f"'{prompt}'에 해당하는 객체를 찾지 못했습니다.", 5000)
            return

        overlayed_image = overlay_masks(self.original_pil_image.copy(), masks)

        self.current_pixmap = self.pil_to_qpixmap(overlayed_image)
        self.update_image_display()
        self.statusBar().showMessage("Segmentation 완료.", 5000)

    def update_image_display(self):
        """현재 QPixmap을 라벨 크기에 맞게 스케일링하여 표시합니다."""
        if self.current_pixmap and not self.current_pixmap.isNull():
            self.image_label.setPixmap(self.current_pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def pil_to_qpixmap(self, pil_img):
        """PIL Image를 QPixmap으로 변환합니다."""
        rgb_image = pil_img.convert("RGB")
        h, w, ch = rgb_image.height, rgb_image.width, 3
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.tobytes(), w, h,
                         bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)

    def resizeEvent(self, event):
        """창 크기 조절 시 이미지를 다시 스케일링합니다."""
        super().resizeEvent(event)
        self.update_image_display()


if __name__ == '__main__':

    print("Starting main...")

    app = QApplication(sys.argv)
    main_win = SegmentationApp()
    main_win.show()
    sys.exit(app.exec())
