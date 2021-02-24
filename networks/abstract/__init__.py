from .modelcatalog import ModelCatalog

# all the method to get the model named as 'get_model_xxxx()'
from ..parts.comem_networks import get_model_comem


def register_models():
    ModelCatalog.register('comem', lambda cfg: get_model_comem(cfg))


register_models()
