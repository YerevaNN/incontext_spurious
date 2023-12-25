import os
import sys
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from src import run_streamlit
from streamlit import runtime

load_dotenv('.env')

@hydra.main(config_path="configs", config_name="visualize", version_base=None)
def start_visualization(cfg: DictConfig):
    run_streamlit(cfg)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

def run_as_streamlit():
    script_name = sys.argv[0]
    script_arguments = " ".join(sys.argv[1:])
    os.system(f"streamlit run {script_name} -- {script_arguments}")

if __name__ == "__main__":
    if runtime.exists():
        start_visualization()
    else:
        run_as_streamlit()