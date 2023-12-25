from omegaconf import DictConfig
import streamlit as st
import torch
import pandas as pd
import altair as alt
import lightning as L

from src.incontextlearner.construction import InContextLearner
from src.incontextlearner.datamodules import ICLDataModule

def plot_chart(size, label_ratio, minority_ratio):
    data = pd.DataFrame(
        {
            "split": [
                "((waterbird on water))",
                "([waterbird on land])",
                "[(landbird on water)]", 
                "[[landbird on land]]",
                ], 
            "count": [
                int(size*label_ratio - int(size*label_ratio*minority_ratio)),
                int(size*label_ratio*minority_ratio), 
                int((size - size*label_ratio)*minority_ratio),
                size - size*label_ratio - int((size  - size*label_ratio)*minority_ratio),
                ], 
            "CONTEXT": ["#"]*4
        }
    )

    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('sum(count)', stack="zero"),
        y='CONTEXT',
        color='split'
    )

    st.altair_chart(chart, theme=None, use_container_width=True)

@st.cache_resource
def init_model(_model_path):
    return InContextLearner.load_from_checkpoint(_model_path)

def model_eval(context_params: dict, _cfg: DictConfig):
    model = init_model(_cfg['model_path'])
    datamodule = ICLDataModule(dict(), dict(), dict(), 
                               batch_size=_cfg["datamodule"]["batch_size"], 
                               num_workers=_cfg["datamodule"]["num_workers"],
                               cache_path=_cfg["data_path"],
                               num_classes=2,
                               num_confounders=2)
    
    datamodule.prepare_data()
    custom_dataloader = datamodule.custom_test_dataloader(context_params)  

    trainer = L.Trainer(
                accelerator=_cfg["accelerator"],
                devices=_cfg['gpu_device'],
                enable_checkpointing=False,
                logger=False
                )

    results = trainer.test(model=model, dataloaders=custom_dataloader)

    results = pd.Series(results[0])

    results = results[list(map(lambda x: "loss" not in x, pd.Series(results).index))]

    st.bar_chart(pd.Series(results))


def run_streamlit(cfg: DictConfig):
    st.set_page_config(
        page_title="InContextLearning",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Evaluation ICL")

    size = st.number_input("Context size:", 0, 200, 100)
    st.subheader("Label ratio:")
    label_ratio = st.slider('', 0.0, 1.0, 0.5, key=0)
    st.subheader("Minority ratio:")
    minority_ratio = st.slider('', 0.0, 1.0, 0.5, key=1)
    
    plot_chart(size, label_ratio, minority_ratio)

    st.button("Compute", on_click=model_eval, 
              args=[dict(
                        size=size,
                        label_ratio=label_ratio, 
                        minority_ratio=minority_ratio,
                    ), 
                    cfg])