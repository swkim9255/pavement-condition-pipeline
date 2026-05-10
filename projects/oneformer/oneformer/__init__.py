from . import modeling

# config
from .config import *
from .metadata import register_inference_metadata

# models
from .oneformer_model import OneFormer

register_inference_metadata()
