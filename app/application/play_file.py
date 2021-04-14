import app.model.Transformer
import app.config as cf
import app.infrastructure.CleanData

import tensorflow as tf

tf.keras.backend.clear_session()

D_MODEL = 128 # 512
NB_LAYERS = 4 # 6
FFN_UNITS = 512 # 2048
NB_PROJ = 8 # 8
DROPOUT_RATE = 0.1 # 0.1



if __name__ == "__main__":

    cd = CleanData(
        input_data_path=cf.europarl_en_path, 
        input_nb_prefix_path=cf.nb_prefix_en_path, 
        output_data_path=cf.europarl_es_path, 
        output_nb_prefix_path=cf.nb_prefix_es_path,
        MAX_LENGTH=20,
        BATCH_SIZE = 64,
        BUFFER_SIZE = 20000
    )

    dataset = cd.get_dataset()

    transformer = Transformer(
        vocab_size_enc=VOCAB_SIZE_EN,
        vocab_size_dec=VOCAB_SIZE_ES,
        d_model=D_MODEL,
        nb_layers=NB_LAYERS,
        FFN_units=FFN_UNITS,
        nb_proj=NB_PROJ,
        dropout_rate=DROPOUT_RATE
    )
