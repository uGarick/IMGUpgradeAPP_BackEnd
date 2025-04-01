# main.py
from utils import pillow_utils
from utils import rembg_utils
#from utils import realesrgan_utils
from utils import filters_utils
from utils import conversion_utils
from utils import style_transfer_utils
from flask import Flask, request, jsonify, send_file
import os
import uuid

app = Flask(__name__)

TEMP_DIR = 'temp'
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.route('/get_styles', methods=['GET'])
def get_styles():
    print("http GET successful")

@app.route('/process_rmbg', methods=['POST'])
def process_rmbg():
    if not request.data:
        return jsonify({"error": "Файл изображения не найден в теле запроса"}), 400

        # Генерируем временное имя для входного изображения
    input_filename = os.path.join(TEMP_DIR, f'{uuid.uuid4().hex}.png')

    # Сохраняем бинарные данные в файл
    with open(input_filename, 'wb') as f:
        f.write(request.data)

    # Генерируем имя для выходного изображения
    output_filename = os.path.join(TEMP_DIR, f'{uuid.uuid4().hex}_cropped.png')

    # Вызов функции удаления фона
    try:
        rembg_utils.remove_background_rembg(input_filename, output_filename)
    except Exception as e:
        return jsonify({"error": f"Ошибка обработки изображения: {str(e)}"}), 500

    # Отправляем результат обратно клиенту
    print("http GET successful")
    return send_file(output_filename, mimetype='image/png')


@app.route('/process', methods=['POST'])
def process_image():
    # Отправляем результат обратно клиенту
    print("http GET successful")



#def main():
    #input_image = "IMGUpgrade_test.jpg"

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
    #style_transfer_utils.neural_style_transfer(input_image, "styles/style.jpg", "test_imgs/styled2_output.jpg", num_steps=300)

if __name__ == '__main__':
    # Запуск сервера на всех интерфейсах, порт можно изменить по необходимости
    app.run(host='0.0.0.0', port=5000)
