from PIL import Image
import numpy as np


def preprocess(obs, normalizer = 1.0, new_size=32):
    img = Image.fromarray(obs)
    img = img.resize((new_size, new_size))
    new_obs = np.array(img)/normalizer
    return new_obs