from PIL import Image

def crop_image(input_path, output_path, crop_box):
    """
    Обрезка изображения по указанным координатам.
    crop_box - кортеж (left, top, right, bottom)
    """
    image = Image.open(input_path)
    cropped = image.crop(crop_box)
    cropped.save(output_path)
    print(f"Изображение обрезано и сохранено как {output_path}")

def flip_image(input_path, output_path, direction='horizontal'):
    """
    Разворот изображения.
    direction: 'horizontal' (зеркальное отражение по горизонтали) или 'vertical' (по вертикали)
    """
    image = Image.open(input_path)
    if direction == 'horizontal':
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == 'vertical':
        flipped = image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("Параметр direction должен быть 'horizontal' или 'vertical'")
    flipped.save(output_path)
    print(f"Изображение развернуто ({direction}) и сохранено как {output_path}")
