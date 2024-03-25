import logging
import os
import numpy as np
import random

from torch.utils.data import Dataset
from src.utils.dataset_helpers import TokenGenerator
from src.datamodules.datasets.waterbirds import CustomizedWaterbirdsDataset as WaterbirdsDataset
from src.utils.dataset_helpers.encoding_transforms import EncodingRotator, IdentityTransform

log = logging.getLogger(__name__)


class WaterbirdsEmbContextsDataset(Dataset):
    """
    A dataset class for the Waterbirds dataset, which handles embeddings and contextual data.
    This class supports either generating new data dynamically or loading pre-existing data from a specified path.

    Attributes:
        root_dir (str): The root directory of the dataset.
        split (str): The type of dataset split ('train', 'val').
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_class_size (int): The size of each class in the context.
        group_proportions:
        are_spurious_tokens_fixed (bool): Flag indicating whether to use fixed spurious tokens.
        are_class_tokens_fixed (bool): Flag indicating whether to use fixed class tokens.
        token_generation_mode (str): Mode of token generation. Accepts 'random' or 'opposite'.
                                     'random' generates tokens with normal distribution,
                                     and 'opposite' generates a pair of tokens where the second is the negative of the first.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'separate_token'(x,c) , 'no_spurious'(x), 'sum'(x+c)
        randomly_change_task (bool): Randomly change the labels during training 0 <-> 1.
        randomly_swap_labels (bool): Randomly change the task during training to predict the background.
        rotate_encodings (bool): Randomly rotate the encodings during training.
        n_rotation_matrices (int): Count of the rotation matrices to use.
        class_dependant_rotate (bool): Rotate the class-encodings independently.
    """

    def __init__(self,
                 root_dir,
                 split,
                 encoding_extractor,
                 data_length,
                 context_class_size,
                 group_proportions,
                 avg_norms_and_encodings,
                 are_spurious_tokens_fixed,
                 are_class_tokens_fixed,
                 token_generation_mode,
                 spurious_setting,
                 randomly_change_task = False,
                 randomly_swap_labels = False,
                 rotate_encodings = False,
                 n_rotation_matrices = None,
                 class_dependant_rotate = False
                 ):
        super(WaterbirdsEmbContextsDataset, self).__init__()

        dataset = WaterbirdsDataset(root_dir,
                                    data_type="encoding",
                                    encoding_extractor=encoding_extractor,
                                    return_labels=True)
        self._split = split
        groups = np.stack([(2 * y + c) for _, y, c, _ in dataset])

        self._train_set = dataset.get_subset("train")
        self._train_groups = groups[self._train_set.indices]

        if self._split == "train":
            self._queries_set = self._train_set
        else:
            self._queries_set = dataset.get_subset(self._split)
        
        self._queries_groups = groups[self._queries_set.indices]

        self._data_length = data_length if self._split == "train" else len(self._queries_set)

        # Dataset parameters
        self._context_class_size = context_class_size
        self._group_proportions = group_proportions

        # Loading tokens data
        tokens_data = np.load(os.path.join(root_dir, avg_norms_and_encodings, "avg_norms", f"{encoding_extractor}_l2.npz"), mmap_mode="r")
        # Convert tokens_data to a dictionary to resolve "Bad CRC-32" error in multi-worker mode.
        tokens_data = {k: tokens_data[k] for k in tokens_data.keys()}

        tokens_generator = TokenGenerator(tokens_data=tokens_data,
                                          are_spurious_tokens_fixed=are_spurious_tokens_fixed,
                                          are_class_tokens_fixed=are_class_tokens_fixed,
                                          token_generation_mode=token_generation_mode)

        self._spurious_tokens_generator, self._class_tokens_generator = tokens_generator()

        self._token_generation_mode = token_generation_mode

        self._spurious_setting = spurious_setting

        self._randomly_swap_labels = randomly_swap_labels
        self._randomly_change_task = randomly_change_task

        if rotate_encodings:
                self._img_encoding_transform = EncodingRotator(n_rotation_matrices, tokens_data["token_len"].item())
                self._class_dependant_rotate = class_dependant_rotate
        else:
            self._img_encoding_transform = IdentityTransform()
            self._class_dependant_rotate = False

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
        tuple: A tuple containing the stacked input sequence of image encodings and tokens,
               along with arrays of spurious labels, class labels, and image indices.
        """
        spurious_tokens = next(self._spurious_tokens_generator)
        class_tokens = next(self._class_tokens_generator)
        context_of_ids, query_of_ids = self._construct_context_and_query_of_ids(
                                                        query_idx=idx if self._split=="val" else None)

        # Process and combine image encodings with tokens
        input_seq, spurious_labels, class_labels, image_indices = self._process_and_combine_encodings(context_of_ids,
                                                                                                      query_of_ids,
                                                                                                      spurious_tokens,
                                                                                                      class_tokens)
        return input_seq, spurious_labels, class_labels, image_indices

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self._data_length

    @staticmethod
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

    def _construct_context_and_query_of_ids(self, query_idx=None):
        """
        Constructs a context dataset and a query instance using identifiers and labels
        generated from two randomly selected categories for machine learning.

        This method uses 'self._dataframe' as the source data and utilizes various class attributes
        like 'categories', 'context_class_size', 'minority_group_proportion', etc.

        Returns:
            tuple: A tuple containing the combined context dataset and the query instance, both primarily based on IDs.
        """

        # Using the class's dataframe
        context_len = 2 * self._context_class_size
        if self._group_proportions == "random":
            class_prop = np.random.uniform(low=0.25, high=0.75)
            minority_prop = np.random.uniform(low=0.25, high=0.5)
            group_proportions = [class_prop * (1 - minority_prop), class_prop * minority_prop,
                                (1 - class_prop) * minority_prop, (1 - class_prop) * (1 - minority_prop)]
        else:
            group_proportions = self._group_proportions
        
        group_counts = [int(context_len * p) for p in group_proportions]

        missing_length = context_len - sum(group_counts) # because of int() we lost a little
        for i in range(missing_length): # creating a context of the exact length *context_len*
            group_counts[np.random.randint(4)] += 1

        context_indices = np.array([], dtype=np.int64)
        for i, group_count in enumerate(group_counts):
            all_group_items = np.where(self._train_groups == i)[0]
            random_items = np.random.choice(all_group_items, group_count, replace=False)
            context_indices = np.concatenate([context_indices, random_items])

        if query_idx is None:
            query_group = np.random.randint(0, 4)
            query_idx = np.random.choice(np.where(self._queries_groups == query_group)[0])

        np.random.shuffle(context_indices)

        context_labels = np.array([self._train_set[idx][1] for idx in context_indices])
        context_spurious_labels = np.array([self._train_set[idx][2] for idx in context_indices])
        query_label = self._queries_set[query_idx][1]
        query_spurious_label = self._queries_set[query_idx][2]

        context = list(zip(context_indices, context_spurious_labels, context_labels))
        query = (query_idx, query_spurious_label, query_label)

        return context, query

    def _process_and_combine_encodings(self, context_of_ids, query_of_ids, spurious_tokens, class_tokens):
        """
        Processes and combines image encodings with spurious and class tokens for the given context and query IDs.

        Args:
            context_of_ids (list): List of tuples containing image IDs, spurious labels, and class labels for the context images.
            query_of_ids (tuple): Tuple containing image ID, spurious label, and class label for the query image.
            spurious_tokens (dict): Dictionary mapping spurious labels to tokens.
            class_tokens (dict): Dictionary mapping class labels to tokens.

        Returns:
            tuple: A tuple containing the stacked input sequence of image encodings and tokens,
                   along with arrays of spurious labels, class labels, and image indices.
        """
        input_seq = []
        image_indices = []
        spurious_labels = []
        class_labels = []

        for image_id, spurious_label, class_label in (context_of_ids + [query_of_ids]):
            image_indices.append(image_id)
            spurious_labels.append(spurious_label)
            class_labels.append(class_label)

            image_enc, *_ = self._queries_set[image_id] if image_id == query_of_ids[0] else self._train_set[image_id]
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
                raise ValueError(f"Invalid spurious setting: '{self._spurious_setting}'. Expected 'separate_token', 'sum', or 'no_spurious'.")

        input_seq.pop()  # removing the label of query from input sequence

        input_seq, spurious_labels, class_labels, image_indices = \
            np.stack(input_seq), np.array(spurious_labels), np.array(class_labels), np.array(image_indices)
        
        if self._split == "train":
            if self._randomly_change_task and np.random.choice([True, False]):
                spurious_labels, class_labels = class_labels, spurious_labels

            if self._randomly_swap_labels and np.random.choice([True, False]):
                class_labels = -class_labels + 1 

            if self._class_dependant_rotate:
                for label in np.unique(class_label):
                    input_seq[class_labels == label] = self._img_encoding_transform(input_seq[class_labels == label])
            else:
                input_seq = self._img_encoding_transform(input_seq)

        return input_seq, spurious_labels, class_labels, image_indices