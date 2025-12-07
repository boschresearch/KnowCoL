from PIL import Image
Image.MAX_IMAGE_PIXELS = 200000000

def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def get_blanc_img():
    return Image.new('RGB', (256, 256), color='white')

def _load_image_from_path(img_path):
    try:
        return pil_loader(img_path), 1
    except Exception as e:
        return get_blanc_img(), 0
    
    