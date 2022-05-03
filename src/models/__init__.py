# managing relative import

import sys
sys.path.append("./src/models")

from .build_model import seq2seq_cnn_model, mlp_model