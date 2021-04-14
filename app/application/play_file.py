import app.model.Transformer
import app.config as cf
import app.infrastructure.CleanData

import tensorflow as tf

tf.keras.backend.clear_session()


if __name__ == "__main__":

    cd = CleanData(
        input_data_path=cf.europarl_en_path, 
        input_nb_prefix_path=cf.nb_prefix_en_path, 
        output_data_path=cf.europarl_es_path, 
        output_nb_prefix_path=cf.nb_prefix_es_path,
        MAX_LENGTH=cf.MAX_LENGTH,
        BATCH_SIZE=cf.BATCH_SIZE,
        BUFFER_SIZE=cf.BUFFER_SIZE
    )

    dataset = cd.get_dataset()

    transformer = Transformer(
        vocab_size_enc=cf.VOCAB_SIZE_EN,
        vocab_size_dec=cf.VOCAB_SIZE_ES,
        d_model=cf.D_MODEL,
        nb_layers=cf.NB_LAYERS,
        FFN_units=cf.FFN_UNITS,
        nb_proj=cf.NB_PROJ,
        dropout_rate=cf.DROPOUT_RATE
    )

    checkpoint_path = cf.CHECKPOINT_PATH

    ckpt = tf.train.Checkpoint(
        transformer=transformer,
        optimizer=optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Último checkpoint restaurado!!")
