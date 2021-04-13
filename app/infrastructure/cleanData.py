import re
import tensorflow as tf
import tensorflow_datasets as tfds
import app.config as cf
from getData import *


class CleanData :
    """
    cleans text data from raw file
    """

    def __init__(self, data_path, nb_prefix_path) :
        self.data_path = data_path
        self.nb_prefix_path = nb_prefix_path
        self.corpus = self.get_cleaned_corpus()


    @classmethod
    def get_data(cls, path):
        with open(path, mode = "r", encoding = "utf-8") as f:
            data_text = f.read()
        return data_text


    def get_non_breaking_prefix(self) :
        prefix_text = self.get_data(self.nb_prefix_path) 
        non_breaking_prefix = prefix_text.split("\n")
        non_breaking_prefix = [' ' + pref + '.' for pref in non_breaking_prefix]
        return non_breaking_prefix

    
    def get_cleaned_corpus(self) :
        corpus = self.get_data(self.data_path) 
        for prefix in self.get_non_breaking_prefix():
            corpus = corpus.replace(prefix, prefix + '$$$')
        corpus = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus)
        corpus = re.sub(r"\.\$\$\$", '', corpus)
        corpus = re.sub(r"  +", " ", corpus)
        corpus = corpus.split('\n')
        return corpus
        

    def tokenize(self) :
        st_enc = tfds.deprecated.text.SubwordTextEncoder
        tokenizer = st_enc.build_from_corpus(
            self.corpus, target_vocab_size=2**13
        )
        return tokenizer


    def get_input_output(self) :
        tokenizer = self.tokenize()
        VOCAB_SIZE = tokenizer.vocab_size + 2
        inputs = [
            [VOCAB_SIZE-2] + tokenizer.encode(sentence) + [VOCAB_SIZE-1] for sentence in self.corpus
        ]
        return inputs

    
    def remove_long_sentences(self, MAX_LENGTH) :
        inputs = get_input_output()
        idx_to_remove = [count for count, sent in enumerate(inputs) if len(sent) > MAX_LENGTH]
        for idx in reversed(idx_to_remove):
            del inputs[idx]
            del outputs[idx]
        seq = tf.keras.preprocessing.sequence
        inputs = seq.pad_sequences(inputs, value=0, padding='post', maxlen=MAX_LENGTH)
        return inputs

    @classmethod
    def get_datasets(self, inputs, outputs, BATCH_SIZE, BUFFER_SIZE) :
        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        dataset = dataset.cache()
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)



if __name__ == "__main__":
    europarl_en_path = "data/es-en/europarl-v7.es-en.en"
    europarl_es_path = "data/es-en/europarl-v7.es-en.es"
    nb_prefix_en_path = "data/nonbreaking_prefix.en"
    nb_prefix_es_path = "data/nonbreaking_prefix.es"

    cd = CleanData(europarl_en_path, nb_prefix_en_path)
    test = cd.get_input_output()

    print(test)


