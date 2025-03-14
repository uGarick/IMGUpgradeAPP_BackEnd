from PIL import Image

def convert_image_format(input_path, output_path, target_format):
    """
    Конвертация изображения в указанный формат.
    :param target_format: например, 'JPEG', 'PNG', 'BMP' и т.д.
    """
    image = Image.open(input_path)
    image.save(output_path, format=target_format)
    print(f"Изображение конвертировано в формат {target_format} и сохранено как {output_path}")
