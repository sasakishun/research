import logging

from keras.models import load_model
from keras_compressor.compressor import compress
import config_mnist as cf

logging.basicConfig(
    level=logging.INFO,
)

model = load_model('./model_raw.h5')
model = compress(model, 7e-1)
model.save('./model_compressed.h5')
