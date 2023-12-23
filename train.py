import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from src import run_pipeline

load_dotenv('.env')

@hydra.main(config_path="configs", config_name="train", version_base=None)
def run_training(cfg: DictConfig):
    n_repeat = cfg.n_repeat

    for i in range(n_repeat):
        get_results = run_pipeline(cfg.experiment_settings)

        print(get_results)

if __name__ == "__main__":
    run_training()