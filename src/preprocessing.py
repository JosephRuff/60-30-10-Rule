import requests
from PIL import Image
from io import BytesIO
import numpy as np
from skimage import color

def get_image_data(source, max_pixels=10000000):
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")
    
    imgwidth, imgheight = img.size
    pixelcount = imgwidth * imgheight
    
    if pixelcount > max_pixels:
        scale = 1/np.sqrt(pixelcount / max_pixels)
        imgwidth = max(int(np.floor(imgwidth*scale)), 1)
        imgheight = max(int(np.floor(imgheight*scale)), 1)
        img = img.resize((imgwidth, imgheight), Image.LANCZOS)
        
    return img

def to_rgb_pixels(img):
    arr = np.array(img).astype(np.float32) / 255.0
    pixels = arr.reshape(-1,3)

    return pixels

def to_hsv_pixels(img):
    arr = np.array(img).astype(np.float32) / 255.0
    pixels = arr.reshape(-1,3)
    pixels = color.rgb2hsv(pixels)
    
    return pixels

def to_lab_pixels(img):
    arr = np.array(img).astype(np.float32) / 255.0
    pixels = arr.reshape(-1,3)
    pixels = color.rgb2lab(pixels)
    
    return pixels
