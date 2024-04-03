from abc import ABC, abstractmethod
import logging
import os
import numpy as np
import random

from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def _generate_spurious_labels(primary_label, secondary_label, class_size, minority_proportion,
                              extra_token=False):
    """
    Generates spurious labels based on given labels and proportions.

    Args:
        primary_label (int): The primary label for the majority of tokens.
        secondary_label (int): The secondary label for the minority of tokens.
        class_size (int): The total number of examples in a class.
        minority_proportion (float): The proportion of the minority group in the class.
        extra_token (bool): Whether to add an extra token. Default is False.

    Returns:
        list: A list of spurious labels.
    """
    majority_count = int(class_size * (1 - minority_proportion))
    minority_count = class_size - majority_count

    spurious_tokens = [primary_label] * majority_count + \
                      [secondary_label] * minority_count

    if extra_token:
        spurious_tokens.append(
            random.choice([primary_label, secondary_label]))

    return spurious_tokens


class BaseEmbContextsDataset(Dataset, ABC):
    """Base class for in-context learning datasets."""

    def __init__(self,
                 dataset_path,
                 encoding_extractor,
                 data_length,
                 context_class_size,
                 minority_group_proportion,
                 token_generation_mode,
                 spurious_setting,
                 rotate_encodings=False,
                 class_dependant_rotate=None,
                 saved_data_path=None):
        """
        Args:

        dataset_path (str): The path to the dataset directory.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_class_size (int): The size of each class in the context.
        minority_group_proportion (float): The proportion of the minority group in the context per class.
        token_generation_mode (str): Mode of token generation. Accepts 'random' or 'opposite'.
                                     'random' generates tokens with normal distribution,
                                     and 'opposite' generates a pair of tokens where the second is the negative of the first.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'separate_token'(x,c) , 'no_spurious'(x), 'sum'(x+c)
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        class_dependant_rotate (bool):  Flag decides whether image embedding rotations are class-specific.
        saved_data_path (str or None): Path for loading data; if None, new data is generated.
        """
        super(BaseEmbContextsDataset, self).__init__()

        # Prepare encodings and data files
        encodings_data = np.load(os.path.join(dataset_path, encoding_extractor, "combined.npz"))
        self._encodings = encodings_data["encodings"]
        self._encodings_indices_map = encodings_data["indices_map"]

        self._data_length = data_length

        # Dataset parameters
        self._context_class_size = context_class_size
        self._minority_group_proportion = minority_group_proportion
        self._token_generation_mode = token_generation_mode
        self._spurious_setting = spurious_setting

        if rotate_encodings:
            self._class_dependant_rotate = class_dependant_rotate
        else:
            self._class_dependant_rotate = False

        self._saved_data_path = saved_data_path

    @abstractmethod
    def _sample_k_examples(self, category: str, k: int) -> list[int]:
        pass

    def img_encoding_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    @property
    @abstractmethod
    def categories(self):
        pass

    @property
    @abstractmethod
    def spurious_tokens_generator(self):
        pass

    @property
    @abstractmethod
    def class_tokens_generator(self):
        pass

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index. If 'self._saved_data_path' is None,
        it generates a new data item. If 'self._saved_data_path' is not None, it loads the data item from
        the saved path.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
        tuple: A tuple containing the stacked input sequence of image encodings and tokens,
               along with arrays of spurious labels, class labels, and image indices.
        """
        if self._saved_data_path is None:
            spurious_tokens = next(self.spurious_tokens_generator)
            class_tokens = next(self.class_tokens_generator)
            context_of_ids, query_of_ids = self._construct_context_and_query_of_ids()
        else:
            data = np.load(os.path.join(self._saved_data_path, f"{idx}.npz"))
            spurious_tokens = data[f"{self._token_generation_mode}_spurious_tokens"]
            class_tokens = data[f"{self._token_generation_mode}_class_tokens"]
            instance = data["instance"]
            context_of_ids = [tuple(x) for x in instance[:-1]]
            query_of_ids = tuple(instance[-1])

        # Process and combine image encodings with tokens
        input_seq, spurious_labels, class_labels, image_indices = self._process_and_combine_encodings(
            context_of_ids, query_of_ids, spurious_tokens, class_tokens)
        return input_seq, spurious_labels, class_labels, image_indices

    def __len__(self) -> int:
        return self._data_length

    def _construct_context_and_query_of_ids(self):
        """
        Constructs a context dataset and a query instance using identifiers and labels
        generated from two randomly selected categories for machine learning.

        Returns:
            tuple: A tuple containing the combined context dataset and the query instance, both primarily based on IDs.
        """

        # Randomly selecting two categories
        if isinstance(self.categories, tuple):  # in case of inner_val-outer set
            category1, category2 = (np.random.choice(x, size=1)[0] for x in self.categories)
        else:
            category1, category2 = np.random.choice(self.categories, size=2, replace=False)

        # Sampling examples from each category
        cat1_indices = self._sample_k_examples(category1, self._context_class_size)
        cat2_indices = self._sample_k_examples(category2, self._context_class_size + 1)

        # Shuffling class and spurious labels
        class_labels, spurious_labels = [0, 1], [0, 1]
        random.shuffle(class_labels)
        random.shuffle(spurious_labels)
        cat1_class_label, cat2_class_label = class_labels
        spurious_label1, spurious_label2 = spurious_labels

        # Generating spurious labels for each category
        cat1_spurious_labels = _generate_spurious_labels(spurious_label1, spurious_label2,
                                                         self._context_class_size,
                                                         self._minority_group_proportion)
        cat2_spurious_labels = _generate_spurious_labels(spurious_label2, spurious_label1,
                                                         self._context_class_size,
                                                         self._minority_group_proportion, extra_token=True)

        # Preparing query example
        query_image_id = cat2_indices.pop()
        query_spurious_label = cat2_spurious_labels.pop()

        # Ensure the class labels are replicated to match the number of indices
        cat1_class_labels = [cat1_class_label] * len(cat1_indices)
        cat2_class_labels = [cat2_class_label] * len(cat2_indices)

        # Pair indices with their respective labels
        cat1_examples = list(zip(cat1_indices, cat1_spurious_labels, cat1_class_labels))
        cat2_examples = list(zip(cat2_indices, cat2_spurious_labels, cat2_class_labels))

        # Combine and shuffle the examples from both categories
        context = cat1_examples + cat2_examples
        random.shuffle(context)

        # Create a query tuple
        query = (query_image_id, query_spurious_label, cat2_class_label)

        return context, query

    def _prepare_image_encodings(self, context_of_ids, query_of_ids):
        """
        Transforms image encodings based on their labels using the appropriate encoding transformer.

        Args:
            context_of_ids (list): List of tuples for the context images containing IDs, spurious labels, and class labels.
            query_of_ids (tuple): Tuple for the query image containing ID, spurious label, and class label.

        Returns:
            np.ndarray: The transformed image encodings.
        """
        img_encodings, labels = [], []
        for image_id, spurious_label, class_label in (context_of_ids + [query_of_ids]):
            img_encodings.append(self._encodings[self._encodings_indices_map[image_id]])
            labels.append(class_label)
        img_encodings, labels = np.stack(img_encodings), np.array(labels)

        if self._class_dependant_rotate:
            for label in np.unique(labels):
                img_encodings[labels == label] = self.img_encoding_transform(img_encodings[labels == label])
        else:
            img_encodings = self.img_encoding_transform(img_encodings)

        return img_encodings

    def _process_and_combine_encodings(self, context_of_ids, query_of_ids, spurious_tokens, class_tokens):
        """
        Processes and combines image encodings with spurious and class tokens for the given context and query IDs.

        Args:
            context_of_ids (list): List of tuples containing image IDs, spurious labels, and class labels for the context images.
            query_of_ids (tuple): Tuple containing image ID, spurious label, and class label for the query image.
            spurious_tokens (np.ndarray): numpy array mapping spurious labels to tokens.
            class_tokens (np.ndarray): numpy array mapping class labels to tokens.

        Returns:
            tuple: A tuple containing the stacked input sequence of image encodings and tokens,
                   along with arrays of spurious labels, class labels, and image indices.
        """
        input_seq = []
        image_indices = []
        spurious_labels = []
        class_labels = []

        img_encodings = self._prepare_image_encodings(context_of_ids, query_of_ids)

        for image_enc, (image_id, spurious_label, class_label) in zip(img_encodings, (context_of_ids + [query_of_ids])):
            image_indices.append(image_id)
            spurious_labels.append(spurious_label)
            class_labels.append(class_label)

            class_token = class_tokens[class_label]

            if self._spurious_setting == 'separate_token':
                spurious_token = spurious_tokens[spurious_label]
                input_seq += [image_enc, spurious_token, class_token]
            elif self._spurious_setting == 'sum':
                spurious_token = spurious_tokens[spurious_label]
                input_seq += [image_enc + spurious_token, class_token]
            elif self._spurious_setting == 'no_spurious':
                input_seq += [image_enc, class_token]
            else:
                raise ValueError(f"Invalid spurious setting: '{self._spurious_setting}'."
                                 f"Expected 'separate_token', 'sum', or 'no_spurious'.")

        input_seq.pop()  # removing the label of query from input sequence

        return np.stack(input_seq), np.array(spurious_labels), np.array(class_labels), np.array(image_indices)