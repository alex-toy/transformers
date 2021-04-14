import app.model.Transformer
import app.config as cf
import app.infrastructure.CleanData as CleanData
import app.model.CustomSchedule as CustomSchedule
import run_model

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

    leaning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(
        leaning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    ckpt = tf.train.Checkpoint(
        transformer=transformer,
        optimizer=optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(ckpt, cf.CHECKPOINT_PATH, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Last checkpoint restored!!")

    
    run_model(dataset)
