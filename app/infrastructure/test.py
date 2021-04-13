import app.config as cf
from getData import *

europarl_en = get_data_text(cf.europarl_en_path)
print(europarl_en[:100])
