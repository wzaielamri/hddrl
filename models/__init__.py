from ray.rllib.models import ModelCatalog
from models.fcnet_glorot_uniform_init import FullyConnectedNetwork_GlorotUniformInitializer
from models.fcnet_glorot_uniform_init_lstm import FullyConnectedNetwork_GlorotUniformInitializer_LSTM


ModelCatalog.register_custom_model(
    "fc_glorot_uniform_init", FullyConnectedNetwork_GlorotUniformInitializer)
ModelCatalog.register_custom_model(
    "fc_glorot_uniform_init_lstm", FullyConnectedNetwork_GlorotUniformInitializer_LSTM)
