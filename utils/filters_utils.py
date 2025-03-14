from PIL import Image, ImageFilter, ImageOps

def apply_grayscale(input_path, output_path):
    """
    Преобразование изображения в оттенки серого.
    """
    image = Image.open(input_path)
    gray_image = ImageOps.grayscale(image)
    gray_image.save(output_path)
    print(f"Изображение преобразовано в оттенки серого и сохранено как {output_path}")

def apply_sepia(input_path, output_path):
    """
    Применение эффекта сепии к изображению.
    """
    image = Image.open(input_path)
    width, height = image.size
    pixels = image.load()  # доступ к пикселям

    for py in range(height):
        for px in range(width):
            r, g, b = image.getpixel((px, py))
            # Алгоритм для эффекта сепии
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            pixels[px, py] = (min(255, tr), min(255, tg), min(255, tb))
    image.save(output_path)
    print(f"Изображение обработано с эффектом сепии и сохранено как {output_path}")

def apply_blur(input_path, output_path, radius=2):
    """
    Применение размытия к изображению (Gaussian Blur).
    """
    image = Image.open(input_path)
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred_image.save(output_path)
    print(f"Изображение с размытием (радиус {radius}) сохранено как {output_path}")
