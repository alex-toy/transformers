import tensorflow as tf
from tensorflow.keras import layers
#import tensorflow_datasets as tfds


def get_data_text(path) :
    with open(path, mode = "r", encoding = "utf-8") as f:
        data_text = f.read()
    return data_text



if __name__ == "__main__":
    europarl_en_path = "data/es-en/europarl-v7.es-en.en"
    europarl_es_path = "data/es-en/europarl-v7.es-en.es"
    nb_prefix_en_path = "data/nonbreaking_prefix.en"
    nb_prefix_es_path = "data/nonbreaking_prefix.es"

    europarl_en = get_data_text(europarl_en_path)
    europarl_es = get_data_text(europarl_es_path)
    nb_prefix_en = get_data_text(nb_prefix_en_path)
    nb_prefix_es = get_data_text(nb_prefix_es_path)

    print(nb_prefix_es[:100])