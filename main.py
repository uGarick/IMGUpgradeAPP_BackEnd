# main.py
from utils import pillow_utils
from utils import rembg_utils
#from utils import realesrgan_utils
from utils import filters_utils
from utils import conversion_utils
from utils import style_transfer_utils

def main():
    input_image = "IMGUpgrade_test.jpg"

    # Обрезка изображения
    #cropped_image = "test_imgs/cropped.jpg"
    #pillow_utils.crop_image(input_image, cropped_image, (100, 100, 400, 400))

    # Разворот изображения
    #flipped_image = "test_imgs/flipped.jpg"
    #pillow_utils.flip_image(input_image, flipped_image, direction='horizontal')

    # Удаление фона (rembg)
    #output_rembg = "test_imgs/output_rembg.png"
    #rembg_utils.remove_background_rembg(input_image, output_rembg)

    # Повышение разрешения (Real-ESRGAN)
    #output_enhanced_image = "test_imgs/output_enhanced.jpg"
    #realesrgan_utils.enhance_resolution(input_image, output_enhanced_image)

    # Применение фильтров и эффектов:
    #filters_utils.apply_grayscale(input_image, "test_imgs/grayscale.jpg")
    #filters_utils.apply_sepia(input_image, "test_imgs/sepia.jpg")
    #filters_utils.apply_blur(input_image, "test_imgs/blur.jpg", radius=3)

    # Конвертация формата изображения:
    #output_converted = "test_imgs/output_converted.png"
    #conversion_utils.convert_image_format(input_image, output_converted, "PNG")

    # Нейронная стилизация:
    # Необходимо предоставить изображение контента и стиля
    style_transfer_utils.neural_style_transfer(input_image, "style2.jpg", "test_imgs/styled2_output.jpg", num_steps=300)

if __name__ == "__main__":
    main()
