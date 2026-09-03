import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor


class FeatureDetector:
    def __init__(self, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[Detectors] Инициализация DPT Depth на {self.device}...")

        self.image_processor = DPTImageProcessor.from_pretrained(
            "Intel/dpt-hybrid-midas"
        )
        self.depth_model = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas"
        )
        self.depth_model.to(self.device)
        self.depth_model.eval()

    @torch.inference_mode()
    def get_depth_map(
        self,
        image: Image.Image,
        inject_details: bool = True,
        detail_alpha: float = 0.2,
        blur_radius: int = 15,
        apply_clahe: bool = True,
        clahe_clip: float = 2.0,
        target_max: int = 210,  # Задаем жесткий верхний предел для предотвращения белых пятен
    ) -> Image.Image:
        rgb_image = image.convert("RGB")
        inputs = self.image_processor(
            images=rgb_image, return_tensors="pt"
        ).to(self.device)

        outputs = self.depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        depth_np = prediction.squeeze().cpu().numpy()

        # 1. Линейная нормализация в базовый диапазон [0, 255]
        d_min, d_max = depth_np.min(), depth_np.max()
        depth_uint8 = ((depth_np - d_min) / (d_max - d_min + 1e-8) * 255.0).astype(np.uint8)

        # 2. Локальное усиление градиентов (морщины/глазницы)
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            depth_uint8 = clahe.apply(depth_uint8)

        # 3. Подмешивание High-Pass рельефа
        if inject_details:
            depth_uint8 = self.inject_high_frequency_details(
                depth_np=depth_uint8,
                source_image=rgb_image,
                alpha=detail_alpha,
                blur_radius=blur_radius,
            )

        # 4. Линейное сжатие всего диапазона (Scale), чтобы убрать клиппинг БЕЗ срезания вершин
        # Вместо clip() мы делим на 255 и умножаем на target_max (210)
        depth_float = depth_uint8.astype(np.float32)
        depth_scaled = (depth_float / 255.0) * float(target_max)

        return Image.fromarray(depth_scaled.astype(np.uint8))

    @staticmethod
    def inject_high_frequency_details(
        depth_np: np.ndarray,
        source_image: Image.Image,
        alpha: float = 0.18,
        blur_radius: int = 21,
    ) -> np.ndarray:
        """Извлечение мелкого рельефа из RGB (High-Pass) с маскированием фона."""
        gray_source = cv2.cvtColor(
            np.array(source_image), cv2.COLOR_RGB2GRAY
        ).astype(np.float32)

        if gray_source.shape[:2] != depth_np.shape[:2]:
            gray_source = cv2.resize(
                gray_source, (depth_np.shape[1], depth_np.shape[0])
            )

        if blur_radius % 2 == 0:
            blur_radius += 1

        # 1. Извлекаем высокие частоты
        low_freq = cv2.GaussianBlur(gray_source, (blur_radius, blur_radius), 0)
        high_freq = gray_source - low_freq

        # 2. Мягкая маска: подмешиваем только на объект, игнорируя темный фон
        object_mask = (depth_np > 15).astype(np.float32)

        # 3. Накладываем детали
        enhanced_depth = depth_np.astype(np.float32) + (
            high_freq * alpha * object_mask
        )
        return np.clip(enhanced_depth, 0, 255).astype(np.uint8)