# NLP project 1 - Transformer

By Alessio Rea

==============================

You need to have Python 3.8.5 installed for this project

# General explanation

## 1. Purpose of the project

The purpose of the project is to create a translation model based on Transformer architecture. The model can be trained on two corpuses of different languages. The first corpus is said to be input and the second is said to be output. The final model will be able to translate from input corpus to output corpus.


## 2. General organisation of the project

The project is divided into three main parts :

    - infrastructure : low level data cleaning
    - model : creation of the different components of the model
    - application : high level functions to orchestrate the model training and the final user interaction.


## 3. Folders

    ├── README.md
    ├── __init__.py
    ├── app
    │   ├── __init__.py
    │   ├── __init__.pyc
    │   ├── __pycache__
    │   │   ├── __init__.cpython-38.pyc
    │   │   ├── config.cpython-38.pyc
    │   │   └── getData.cpython-38.pyc
    │   ├── application
    │   │   ├── __init__.py
    │   │   ├── __init__.pyc
    │   │   ├── __pycache__
    │   │   │   ├── evaluate.cpython-38.pyc
    │   │   │   ├── run_model.cpython-38.pyc
    │   │   │   ├── train_model.cpython-38.pyc
    │   │   │   └── translate.cpython-38.pyc
    │   │   ├── evaluate.py
    │   │   ├── evaluate.pyc
    │   │   ├── play_file.py
    │   │   ├── train_model.py
    │   │   ├── train_model_step_1.py
    │   │   ├── train_model_step_2.py
    │   │   └── translate.py
    │   ├── config.py
    │   ├── config.pyc
    │   ├── infrastructure
    │   │   ├── CleanData.py
    │   │   ├── __init__.py
    │   │   └── __pycache__
    │   │       ├── CleanData.cpython-38.pyc
    │   │       └── getData.cpython-38.pyc
    │   └── model
    │       ├── CustomSchedule.py
    │       ├── Decoder.py
    │       ├── DecoderLayer.py
    │       ├── Encoder.py
    │       ├── EncoderLayer.py
    │       ├── MultiHeadAttention.py
    │       ├── PositionalEncoding.py
    │       ├── Transformer.py
    │       ├── __init__.py
    │       ├── __pycache__
    │       │   ├── CustomSchedule.cpython-38.pyc
    │       │   ├── Decoder.cpython-38.pyc
    │       │   ├── DecoderLayer.cpython-38.pyc
    │       │   ├── Encoder.cpython-38.pyc
    │       │   ├── EncoderLayer.cpython-38.pyc
    │       │   ├── MultiHeadAttention.cpython-38.pyc
    │       │   ├── PositionalEncoding.cpython-38.pyc
    │       │   ├── Transformer.cpython-38.pyc
    │       │   ├── loss_function.cpython-38.pyc
    │       │   └── scaled_dot_product_attention.cpython-38.pyc
    │       ├── loss_function.py
    │       └── scaled_dot_product_attention.py
    ├── checkpoint
    │   ├── README.md
    │   ├── checkpoint
    │   └── ckpt-58.index
    ├── data
    │   ├── es_en
    │   │   ├── europarl-v7.es-en.en
    │   │   └── europarl-v7.es-en.es
    │   ├── es_en_test
    │   │   ├── europarl-v7.es-en.en
    │   │   └── europarl-v7.es-en.es
    │   ├── nonbreaking_prefix.en
    │   ├── nonbreaking_prefix.es
    │   └── utils.txt
    ├── models
    │   ├── INPUT_VOCAB_SIZE.joblib
    │   ├── OUTPUT_VOCAB_SIZE.joblib
    │   ├── README.md
    │   ├── input_tokenizer.joblib
    │   └── output_tokenizer.joblib
    ├── nb
    │   └── Transformer_para_NLP.ipynb
    ├── output
    │   ├── README.md
    │   ├── inputs.csv
    │   └── outputs.csv
    └── requirements.txt