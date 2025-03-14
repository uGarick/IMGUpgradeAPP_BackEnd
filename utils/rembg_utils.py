from rembg import remove
from PIL import Image
import io


def remove_background_rembg(input_path, output_path):
    input_image = Image.open(input_path)
    # Преобразуем изображение в байты
    buf = io.BytesIO()
    input_image.save(buf, format="PNG")
    buf = buf.getvalue()

    output_bytes = remove(buf)
    output_image = Image.open(io.BytesIO(output_bytes))
    output_image.save(output_path)
    print(f"Фон удален (rembg) и сохранен как {output_path}")
