from abc import ABC, abstractmethod
from typing import Optional, Tuple
import logging
import os

import numpy as np

from torch.utils.data import Dataset
from src.utils.dataset_helpers import EncodingRotator, IdentityTransform, PartlySwapper
from src.utils.dataset_helpers.context_prep_utils import get_context_example_tokens,\
    get_query_example_tokens

log = logging.getLogger(__name__)

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


def _generate_x_spurious_tokens(
        token_len: int,
        avg_norm: float,
        sp_token_generation_mode: str
) -> np.ndarray:

    if sp_token_generation_mode == "random":
        x_spurious_tokens = np.random.randn(2, token_len).astype(np.float32)
    else:  # opposite
        first_row = np.random.randn(1, token_len).astype(np.float32)
        x_spurious_tokens = np.vstack((first_row, -first_row))

    x_spurious_tokens = (x_spurious_tokens / np.linalg.norm(
        x_spurious_tokens, axis=1, keepdims=True)) * avg_norm

    return x_spurious_tokens


class BaseEmbContextsDatasetV2(Dataset, ABC):
    """Base class for V2 datasets."""

    def __init__(self,
                 encoding_extractor: str,
                 data_length: int,
                 context_class_size: int,
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 use_context_as_intermediate_queries: bool = False,
                 rotate_encodings: bool = False,
                 n_rotation_matrices: Optional[int] = None,
                 label_noise_ratio_interval: Optional[list] = None,
                 input_noise_norm_interval: Optional[list] = None,
                 permute_input_dim: bool = False,
                 ask_context_prob: Optional[float] = None,
                 swapping_minority_proportion_context: Optional[float] = None,
                 swapping_minority_proportion_query: Optional[float] = None,
                 points_to_swap_range: Optional[list] = None,
                 ):
        """
        Arguments:
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_class_size (int): The size of each class in the context.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
        sp_token_generation_mode (str): Specifies whether the representations of two spurious labels should be
                                        'opposite' or 'random'.
        use_context_as_intermediate_queries (bool): Whether intermediate queries should be the context examples.
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
        label_noise_ratio_interval (list or None): Interval for the ratio of label noise. 
                                If None, no label noise is added.
        input_noise_norm_interval (list or None): Interval for the norm of Gaussian noise.
                                If None, no Gaussian noise is added to representations.
        permute_input_dim (bool): Determines if image encodings are permuted. 
                                True enables permutation, while False bypasses it.
        ask_context_prob (float or None): If specified, defines the probability with which a query is set to be one
                                          of previous context examples.
        swapping_minority_proportion_context (float): The proportion of the minority group's to create via swapping in context.
        swapping_minority_proportion_query (float): The proportion of the minority group's to create via swapping in queries.
        points_to_swap_range (list): A list containing the range of the number of points to swap in the selected vectors.
        """
        super(BaseEmbContextsDatasetV2, self).__init__()

        self._encoding_extractor = encoding_extractor
        self._data_length = data_length
        self._context_class_size = context_class_size
        self._spurious_setting = spurious_setting
        self._sp_token_generation_mode = sp_token_generation_mode
        self._use_context_as_intermediate_queries = use_context_as_intermediate_queries
        self._rotate_encodings = rotate_encodings
        self._n_rotation_matrices = n_rotation_matrices
        self._label_noise_ratio_interval = label_noise_ratio_interval
        self._input_noise_norm_interval = input_noise_norm_interval
        self._permute_input_dim = permute_input_dim
        self._ask_context_prob = ask_context_prob

        # Loading tokens data
        if self._encoding_extractor in ["stroberta", "bert", "roberta"]:
            token_data_path = os.path.join(
                os.environ.get('DATA_ROOT_DIR'),
                'multinli/avg_norms',
                f"{encoding_extractor}_l2.npz")
        else:
            token_data_path = os.path.join(
                os.environ.get('DATA_ROOT_DIR'),
                'inaturalist2017/avg_norms',
                f"{encoding_extractor}_l2.npz")

        tokens_data = np.load(token_data_path)
        self._token_len = np.copy(tokens_data["token_len"])
        self._opposite_class_tokens = np.copy(tokens_data["opposite_class_tokens"])
        self._opposite_spurious_tokens_2 = np.copy(tokens_data["opposite_spurious_tokens_2"])
        self._random_class_tokens = np.copy(tokens_data["random_class_tokens"])
        self._avg_norm = np.copy(tokens_data["avg_norm"])

        self._use_context_as_intermediate_queries = use_context_as_intermediate_queries

        if rotate_encodings:
            self._img_encoding_transform = EncodingRotator(n_rotation_matrices, tokens_data["token_len"].item())
        else:
            self._img_encoding_transform = IdentityTransform()

        if spurious_setting in ['swap_erm', 'swap_dro']:
            self._partly_swapper = PartlySwapper(
                swapping_minority_proportion_context,
                swapping_minority_proportion_query,
                points_to_swap_range)
        else:
            self._partly_swapper = None

    def __getitem__(self, idx) -> Tuple[np.ndarray, Examples, Examples, np.ndarray]:
        """Returns a dataset example given the example index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns: A tuple (input_seq, context, queries, query_indices), where input_seq is a np.ndarray of shape
            (total_seq_len, dim) that is to be fed ta a V2 transformer. Context and queries are np.ndarrays of shape
            (2 * context_class_size, 3) describing context/query examples with (id, spurious, class) tuples.
            query_indices specifies positions of input_seq corresponding to query tokens.
        """

        # get spurious and class tokens
        x_spurious_tokens = _generate_x_spurious_tokens(
            token_len=self._token_len,
            avg_norm=self._avg_norm,
            sp_token_generation_mode=self._sp_token_generation_mode
        )
        c_spurious_tokens = self._opposite_spurious_tokens_2
        class_tokens = self._opposite_class_tokens

        # The following code block produces 2*self._context_class_size context and queries examples.
        # Some queries will be (0, 0, 0) implying that the query in the corresponding position is empty.
        # These empty queries are mapped to the next context example and this mapping is specified in
        # the `query_to_context_map` of length 2*self._context_class_size. The value -1 in this array
        # means that the corresponding query is not replaced.

        if self._use_context_as_intermediate_queries:
            context, last_query = self._generate_context_and_queries(
                num_context_examples=2 * self._context_class_size,
                num_query_examples=1,
            )
            queries = np.zeros_like(context)
            queries[-1:] = last_query
            query_to_context_map = np.concatenate([
                np.arange(1, len(context)),  # points to next context example
                np.array([-1]),              # points to the single query example
            ], axis=0)
        else:
            context, queries = self._generate_context_and_queries(
                num_context_examples=2 * self._context_class_size,
                num_query_examples=2 * self._context_class_size,
            )
            query_to_context_map = -np.ones(len(context), dtype=np.int32)  # keeping sampled queries

        # if specified, occasionally queries past context examples to speed up learning induction heads
        if self._ask_context_prob is not None:
            for q_idx in range(len(context)):
                if np.random.rand() < self._ask_context_prob:
                    c_idx = np.random.randint(q_idx + 1)  # selecting one of previous context examples
                    query_to_context_map[q_idx] = c_idx

        # construct context and query image encodings
        # update `queries`
        context_img_encodings = self._prepare_context_image_encodings(context)
        query_img_encodings = self._prepare_query_image_encodings(queries)
        for q_idx in range(len(context)):
            c_idx = query_to_context_map[q_idx]
            if c_idx != -1:
                query_img_encodings[q_idx] = context_img_encodings[c_idx]
                queries[q_idx] = context[c_idx]

        # do data augmentations if needed
        context_img_encodings, query_img_encodings = self._maybe_rotate_embeddings(
            context_img_encodings=context_img_encodings,
            query_img_encodings=query_img_encodings)
        
        context_img_encodings, context[:, 1], query_img_encodings, queries[:, 1] = self._maybe_partly_swap_points(
            context_img_encodings=context_img_encodings,
            context_labels=context[:, 2],
            context_spurs=context[:, 1],
            query_img_encodings=query_img_encodings,
            query_labels=queries[:, 2],
            query_spurs=queries[:, 1],
        )

        context_img_encodings, query_img_encodings = self._maybe_permute_embeddings(
            context_img_encodings=context_img_encodings,
            query_img_encodings=query_img_encodings
        )

        self._maybe_add_label_noise(context)

        context_img_encodings = self._maybe_add_input_noise(
            context_img_encodings=context_img_encodings
        )

        input_seq = []
        query_indices = []
        for i in range(len(context)):
            # Add current context related tokens.
            _, sp, label = context[i]
            input_seq.extend(get_context_example_tokens(
                img_encoding=context_img_encodings[i],
                x_spurious_token=x_spurious_tokens[sp],
                c_spurious_token=c_spurious_tokens[sp],
                class_token=class_tokens[label],
                spurious_setting=self._spurious_setting,
            ))

            # Add current query related token.
            # NOTE: no matter what spurious setting we use, query spurious label
            #       and class label will not get their own tokens.
            _, sp, _ = queries[i]
            query_tokens = get_query_example_tokens(
                img_encoding=query_img_encodings[i],
                x_spurious_token=x_spurious_tokens[sp],
                spurious_setting=self._spurious_setting)
            assert len(query_tokens) == 1
            query_indices.append(len(input_seq))
            input_seq.extend(query_tokens)

        input_seq = np.stack(input_seq)

        return input_seq, context, queries, np.array(query_indices)

    def __len__(self):
        """Returns the total number of ICL instances."""
        return self._data_length

    @abstractmethod
    def _generate_context_and_queries(
            self,
            num_context_examples: int,
            num_query_examples: int,
    ) -> Tuple[Examples, Examples]:
        """Should sample context and query examples with configured group proportions."""
        pass

    @abstractmethod
    def _prepare_context_image_encodings(
            self,
            context: Examples,
    ) -> np.ndarray:
        """Should return a np.ndarray of stacked embeddings of context examples."""
        pass

    @abstractmethod
    def _prepare_query_image_encodings(
            self,
            queries: Examples,
    ) -> np.ndarray:
        """Should return a np.ndarray of stacked embeddings of queries."""
        pass

    def _maybe_rotate_embeddings(
            self,
            context_img_encodings: np.ndarray,
            query_img_encodings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        joint = np.concatenate([context_img_encodings, query_img_encodings],
                               axis=0)
        joint = self._img_encoding_transform(joint)
        n = context_img_encodings.shape[0]
        return joint[:n], joint[n:]
    
    def _maybe_partly_swap_points(
            self,
            context_img_encodings: np.ndarray,
            context_labels: np.ndarray,
            context_spurs: np.ndarray,
            query_img_encodings: np.ndarray,
            query_labels: np.ndarray,
            query_spurs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._partly_swapper is None:
            return context_img_encodings, context_spurs, query_img_encodings, query_spurs
            
        return self._partly_swapper(
            context_img_encodings, 
            context_labels, 
            context_spurs,
            query_img_encodings,
            query_labels,
            query_spurs,
        )
    
    def _maybe_permute_embeddings(
            self,
            context_img_encodings: np.ndarray,
            query_img_encodings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._permute_input_dim:
            random_permutation = np.random.permutation(context_img_encodings.shape[1])

            context_img_encodings = context_img_encodings[:, random_permutation]
            query_img_encodings = query_img_encodings[:, random_permutation]

        return context_img_encodings, query_img_encodings

    def _maybe_add_label_noise(
            self,
            context: Examples,
    ) -> None:
        if self._label_noise_ratio_interval:
            low, high = self._label_noise_ratio_interval

            flip_p = np.random.uniform(low=low, high=high)
            label_noise_count = int(flip_p * len(context))
            label_noise_indices = np.random.choice(len(context), label_noise_count, replace=False)
            context[label_noise_indices, 2] = 1 - context[label_noise_indices, 2]

    def _maybe_add_input_noise(
            self,
            context_img_encodings: np.ndarray
    ) -> np.ndarray:
        if self._input_noise_norm_interval:
            low, high = self._input_noise_norm_interval

            input_noise_norm = np.random.uniform(low=low, high=high)
            seq_len, input_dim = context_img_encodings.shape

            input_noise = np.random.normal(
                scale=1/np.sqrt(input_dim), size=(seq_len, input_dim),
            ).astype(context_img_encodings.dtype) * input_noise_norm

            context_img_encodings = context_img_encodings + input_noise

        return context_img_encodings
