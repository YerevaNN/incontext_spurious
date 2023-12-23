import lightning as L

from torch.utils.data import DataLoader


def evaluate_model(
        trainer: L.Trainer,
        model: L.LightningModule,
        eval_loaders: dict[str, DataLoader],
        num_groups: int,
        overall_acc_metric_name: str = 'test_accuracy',
        worst_group_acc_metric_name: str = 'test_accuracy_wg',
        group_acc_metric_name_fmt: str = 'test_accuracy_g{}',
) -> dict[str, tuple]:
    results = dict()

    for set_name, eval_loader in eval_loaders.items():
        cur_results = trainer.test(model, eval_loader)[0]

        overall_acc = cur_results[overall_acc_metric_name]
        worst_group_acc = cur_results[worst_group_acc_metric_name]

        group_accuracies = [
            cur_results[group_acc_metric_name_fmt.format(idx)]
            for idx in range(num_groups)
        ]

        results[set_name] = (overall_acc, worst_group_acc, group_accuracies)

    return results
