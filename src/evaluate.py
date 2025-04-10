from collections.abc import Iterable
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

from hydra.utils import instantiate, get_class
from omegaconf import DictConfig

import torch.serialization

from src.utils.helper_functions import get_allowed_classes

matplotlib.use('Agg')
log = logging.getLogger(__name__)


def evaluate(config: DictConfig):
    # Instantiate model from configuration
    log.info(f"Instantiating model <{config.model._target_}>")
    model_class = get_class(config.model._target_)
    del config.model._target_  # Remove _target_ key before instantiation

    with torch.serialization.safe_globals(get_allowed_classes()):
        # Specify map_location='cpu' to initially load the model on the CPU, overriding the default behavior of loading
        # it on the GPU it was trained on. This provides flexibility for the model to be moved to a GPU as per the
        # configuration settings of the trainer.
        model = model_class.load_from_checkpoint(
            config.checkpoint_path,
            **instantiate(config.model),
            map_location='cpu')

    # Instantiate trainer from configuration
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = instantiate(config.trainer)

    context_class_sizes = list(config.datamodule.context_class_size) \
        if isinstance(config.datamodule.context_class_size, Iterable) \
        else [config.datamodule.context_class_size]

    val_sets = [f"val_{x}" for x in config.datamodule.val_sets] if config.datamodule.val_sets else ['val']
    list_of_results = run_evaluations_with_repetitions(trainer, model,
                                                       datamodule_config=config.datamodule,
                                                       context_class_sizes=context_class_sizes,
                                                       val_sets=val_sets,
                                                       n_repeat=config.n_repeat)

    transform_and_filter_results(list_of_results)
    combined_results = aggregate_results_by_set_and_class_size(list_of_results, val_sets, context_class_sizes)

    for set_name in val_sets:
        results_mean_sem = combined_results[set_name]

        create_and_save_results_dataframe(results_mean_sem, context_class_sizes,
                                          title=set_name,
                                          filename=f"{set_name}.csv")

        if len(context_class_sizes) > 1:
            title = f"{config.datamodule.name}, {set_name} | {config.spurious_setting} | {config.aim_hash}"
            plot_results_and_save_figure(results_mean_sem,
                                         title=title,
                                         filename=f"{set_name}.png")


def run_evaluations_with_repetitions(trainer, model, datamodule_config, context_class_sizes, val_sets, n_repeat):
    """
    Runs the evaluation of the model multiple times over different validation sets and context class sizes to ensure
    statistical reliability of the results.

    Args:
        trainer: The instantiated trainer to use for evaluation.
        model: The model to be evaluated.
        datamodule_config: Configuration for the data module.
        context_class_sizes (list): List of context class sizes for evaluation.
        val_sets (list): List of validation sets to evaluate on.
        n_repeat (int): Number of repetitions for each evaluation setup.

    Returns:
        list: A list of dictionaries containing evaluation results for each repetition.
    """
    list_of_results = []
    for repeat_idx in range(n_repeat):
        results = {set_name: [] for set_name in val_sets}

        for context_class_size in context_class_sizes:
            log.info(f"Starting {repeat_idx=} of {context_class_size=}")
            datamodule_config.context_class_size = context_class_size
            # Instantiate data module from configuration
            log.info(f"Instantiating datamodule <{datamodule_config._target_}>")
            datamodule = instantiate(datamodule_config)

            # Validate model using trainer and datamodule
            eval_results = trainer.validate(model, datamodule=datamodule)
            for i, result in enumerate(eval_results):
                set_name = val_sets[i]
                result = {key.split(f"{set_name}_")[-1].split("/")[0]: val for key, val in result.items()}
                # key.split(f"{set_name}_")[-1].split("/")[0] is for 'val_inner_loss_epoch/dataloader_idx_0'-like cases

                results[set_name].append(result)

        list_of_results.append(results)

    return list_of_results


def transform_and_filter_results(list_of_results):
    """
    Filters and transforms the raw results from evaluations, focusing specifically on accuracy metrics.

    Args:
        list_of_results (list): A list containing dictionaries of raw evaluation results.

    Returns:
        None: The function modifies the list_of_results in-place.
    """
    for i, results in enumerate(list_of_results):
        list_of_results[i] = {}
        for set_name, result in results.items():
            result = {metric_name: [res[metric_name]
                                    for res in result] for metric_name in result[0]
                      if "accuracy" in metric_name}

            list_of_results[i][set_name] = result


def aggregate_results_by_set_and_class_size(list_of_results, val_sets, context_class_sizes):
    """
    Aggregates the results from multiple evaluations, calculating the mean and standard error of the mean (SEM)
    for each metric, validation set, and context class size.

    Args:
        list_of_results (list): List of dictionaries containing processed evaluation results.
        val_sets (list): List of validation sets used in the evaluations.
        context_class_sizes (list): List of different context class sizes evaluated.

    Returns:
        dict: A dictionary with keys for each validation set and values containing aggregated results.
    """
    combined_results = {}

    for set_name in val_sets:
        results_mean_sem = {
            metric_name: pd.DataFrame({
                "context_class_size": context_class_sizes,
                'mean': np.mean([combined_res[set_name][metric_name] for combined_res in list_of_results],
                                axis=0),
                'sem': np.nan_to_num(st.sem([combined_res[set_name][metric_name] for combined_res in list_of_results],
                              axis=0)) # nan_to_num handles cases where n_repeat=1, and st.sem returns nans.
            })
            for metric_name in list_of_results[0][set_name]
        }

        combined_results[set_name] = results_mean_sem

    return combined_results


def create_and_save_results_dataframe(results_mean_sem, context_class_sizes, title, filename):
    """
    Creates a Pandas DataFrame from the aggregated results and saves it to a CSV file.

    Args:
        results_mean_sem (dict): Dictionary containing the mean and SEM of the results.
        context_class_sizes (list): List of context class sizes evaluated.
        title (str): Title for the output, used for logging purposes.
        filename (str): Path to the file where the DataFrame will be saved.

    Returns:
        None.
    """
    res = pd.DataFrame(dict(context_class_size=context_class_sizes))

    for metric_name, result_stats in results_mean_sem.items():
        res[metric_name] = result_stats.apply(lambda row: f"{100 * row['mean']:.2f} ± {100 * row['sem']:.2f}",
                                              axis=1)

    print(title)
    print(res)
    print('\n')

    res.to_csv(filename)


def plot_results_and_save_figure(results_mean_sem, title, filename):
    """
    Plots the aggregated results and saves the figure to a file.

    Args:
        results_mean_sem (dict): Dictionary containing the mean and SEM of the results for plotting.
        title (str): Title of the plot.
        filename (str): Path to the file where the plot will be saved.

    Returns:
        None.
    """
    fig, axs = plt.subplots(1, len(results_mean_sem), figsize=(3 * len(results_mean_sem), 4))

    min_value = 1.0
    max_value = 0.0
    for (metric_name, result_stats), ax in zip(results_mean_sem.items(), axs):
        mean_values = result_stats['mean']
        sem_values = result_stats['sem']
        context_class_sizes = result_stats['context_class_size']

        ax.errorbar(context_class_sizes, mean_values, yerr=sem_values, fmt='-o', capsize=5)
        ax.set_title(metric_name)
        ax.set_xticks(context_class_sizes)
        ax.axhline(y=1.0, color='gray', linestyle='--')

        min_value = np.min([np.min(mean_values - sem_values), min_value])
        max_value = np.max([np.max(mean_values + sem_values), max_value])

    for ax in axs:
        ax.set_ylim(min_value - 0.005, max_value + 0.005)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
