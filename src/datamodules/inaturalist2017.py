from torch.utils.data import DataLoader
from src.datamodules.datasets import INaturalist2017Dataset


class INaturalist2017DataModule:
    """
    A data module for handling the iNaturalist 2017 dataset.

    This class facilitates the creation of a DataLoader for the iNaturalist 2017 dataset, allowing for easy
    batching, shuffling, and parallel loading of the data using PyTorch. It encapsulates dataset initialization,
    including resizing and cropping of images, and provides a method to obtain a DataLoader for the dataset.

    Args:
        dataset_path (str): The file path to the dataset directory.
        min_images_per_category (int): The minimum number of images required per category.
        resize_size (int): The size to which the smaller edge of the image will be resized.
        crop_size (int): The size of the square crop to be extracted from the center of the image.
        fully_outer_supercategories (list of str): List of supercategories that should be entirely classified as 'outer'. All categories under these supercategories will be assigned to an 'outer' split.
        fully_inner_supercategories (list of str): List of supercategories that should be entirely classified as 'inner'. All categories under these supercategories will be assigned to an 'inner' split.
        outer_classes_size (float): A decimal representing the proportion of categories within a non-fully classified supercategory to be randomly assigned to the 'outer' split. The rest will be assigned to the 'inner' split.
        inner_val_size (float): A decimal indicating the proportion of data within 'inner' categories to be used as validation data ('inner_val'). The rest of the data in 'inner' categories will be used for training ('inner_train').
        batch_size (int): The number of items to be included in each batch.
        num_workers (int): The number of subprocesses to use for data loading.
    """

    def __init__(self,
                 dataset_path,
                 min_images_per_category,
                 resize_size,
                 crop_size,
                 fully_outer_supercategories,
                 fully_inner_supercategories,
                 outer_classes_size,
                 inner_val_size,
                 batch_size,
                 num_workers,
                 *args, **kwargs):
        # Initialize the INaturalist2017Dataset
        self._dataset = INaturalist2017Dataset(dataset_path,
                                               min_images_per_category,
                                               resize_size,
                                               crop_size,
                                               fully_outer_supercategories,
                                               fully_inner_supercategories,
                                               outer_classes_size,
                                               inner_val_size)

        # Store batch size and number of workers for data loading
        self._batch_size = batch_size
        self._num_workers = num_workers

    def get_dataloader(self):
        """
        Creates and returns a DataLoader for the iNaturalist 2017 dataset.

        Returns:
            DataLoader: A DataLoader instance for the iNaturalist 2017 dataset.
        """
        # Initialize the DataLoader with the dataset, batch size, and number of workers
        dataloader = DataLoader(self._dataset,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers)

        return dataloader
