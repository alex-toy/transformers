import app.model.Transformer

tf.keras.backend.clear_session()

D_MODEL = 128 # 512
NB_LAYERS = 4 # 6
FFN_UNITS = 512 # 2048
NB_PROJ = 8 # 8
DROPOUT_RATE = 0.1 # 0.1

transformer = Transformer(
    vocab_size_enc=VOCAB_SIZE_EN,
    vocab_size_dec=VOCAB_SIZE_ES,
    d_model=D_MODEL,
    nb_layers=NB_LAYERS,
    FFN_units=FFN_UNITS,
    nb_proj=NB_PROJ,
    dropout_rate=DROPOUT_RATE
)

