import sys
from PIL import Image

def pixelate(image_path, output_path, block_size=8):
    img = Image.open(image_path)
    w, h = img.size
    img_small = img.resize((max(1, w // block_size), max(1, h // block_size)), Image.BOX)
    img_pix = img_small.resize((w, h), Image.NEAREST)
    img_pix.save(output_path)

if __name__ == '__main__':
    pixelate('photo.jpg', 'pure_pix.jpg')
