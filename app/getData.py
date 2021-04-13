import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds


path = "data/es-en/europarl-v7.es-en.en"

def get_data_text(path) :
    with open(path, mode = "r", encoding = "utf-8") as f:
        data_text = f.read()
    return data_text



print(get_data_text(path)[:100])