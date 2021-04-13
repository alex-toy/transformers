import app.config as cf
from getData import *

nb_prefix_es = get_data_text(cf.nb_prefix_es_path)
print(nb_prefix_es[:100])

