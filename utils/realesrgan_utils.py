'''
import torch
import cv2
import numpy as np
from realesrgan import RealESRGANer


def enhance_resolution(input_path, output_path, model_path="weights/RealESRGAN_x4plus.pth"):
    """
    Повышает разрешение изображения с использованием Real-ESRGAN.
    :param input_path: путь к входному изображению
    :param output_path: путь для сохранения результата
    :param model_path: путь к модели Real-ESRGAN
    :param scale: коэффициент увеличения (обычно 2, 4, 8)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    model = RealESRGANer(scale=4, device=device, model_path=model_path)
    model.load_weights(model_path, download=True)

    # Загрузка изображения
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Повышение разрешения
    sr_image = model.predict(image)

    # Сохранение результата
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, sr_image)
    print(f"Изображение с повышенным разрешением сохранено как {output_path}")
'''