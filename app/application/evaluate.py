import os
import app.config as cf
from joblib import dump, load

import tensorflow as tf

from app.model.Transformer import Transformer

def evaluate(inp_sentence):
    INPUT_VOCAB_SIZE = load(os.path.join(cf.OUTPUTS_MODELS_DIR, 'INPUT_VOCAB_SIZE.joblib'))
    OUTPUT_VOCAB_SIZE = load(os.path.join(cf.OUTPUTS_MODELS_DIR, 'OUTPUT_VOCAB_SIZE.joblib'))

    input_tokenizer = load(os.path.join(cf.OUTPUTS_MODELS_DIR, 'input_tokenizer.joblib'))


    inp_sentence = [INPUT_VOCAB_SIZE-2] + input_tokenizer.encode(inp_sentence) + [INPUT_VOCAB_SIZE-1]
    enc_input = tf.expand_dims(inp_sentence, axis=0)
    
    output = tf.expand_dims([OUTPUT_VOCAB_SIZE-2], axis=0)

    transformer = Transformer(
        vocab_size_enc=INPUT_VOCAB_SIZE,
        vocab_size_dec=OUTPUT_VOCAB_SIZE,
        d_model=cf.D_MODEL,
        nb_layers=cf.NB_LAYERS,
        FFN_units=cf.FFN_UNITS,
        nb_proj=cf.NB_PROJ,
        dropout_rate=cf.DROPOUT_RATE
    )
    
    for _ in range(cf.MAX_LENGTH):
        predictions = transformer(enc_input, output, False) #(1, seq_length, VOCAB_SIZE_ES)
        
        prediction = predictions[:, -1:, :]
        
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        
        if predicted_id == OUTPUT_VOCAB_SIZE-1:
            return tf.squeeze(output, axis=0)
        
        output = tf.concat([output, predicted_id], axis=-1)
        
    return tf.squeeze(output, axis=0)

