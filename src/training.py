import hydra
from omegaconf import DictConfig

from src.utils import Pipeline

def run_pipeline(cfg_pipeline: DictConfig):
    pipeline_arr = [hydra.utils.instantiate(x) for x in cfg_pipeline.values()]
    pipeline = Pipeline(pipeline_arr)

    res = pipeline()
    return res