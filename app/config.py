import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

europarl_en_path = "data/es-en/europarl-v7.es-en.en"
europarl_es_path = "data/es-en/europarl-v7.es-en.es"
nb_prefix_en_path = "data/nonbreaking_prefix.en"
nb_prefix_es_path = "data/nonbreaking_prefix.es"

OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
INPUTS_FILE = os.path.join(OUTPUTS_DIR, 'inputs.csv')
OUTPUTS_FILE = os.path.join(OUTPUTS_DIR, 'outputs.csv')