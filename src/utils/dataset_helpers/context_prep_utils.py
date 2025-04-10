import numpy as np

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


def prepare_context_or_query(
        cat1_indices: np.ndarray,
        cat2_indices: np.ndarray,
        cat1_spurious_labels: np.ndarray,
        cat2_spurious_labels: np.ndarray,
        cat1_class_label: int,
        cat2_class_label: int) -> Examples:
    """Combines and shuffles list of examples from 2 classes."""
    cat1_class_labels = np.full(shape=(len(cat1_indices),), fill_value=cat1_class_label)
    cat2_class_labels = np.full(shape=(len(cat2_indices),), fill_value=cat2_class_label)
    cat1_examples = np.stack([cat1_indices, cat1_spurious_labels, cat1_class_labels], axis=1)
    cat2_examples = np.stack([cat2_indices, cat2_spurious_labels, cat2_class_labels], axis=1)
    examples = np.concatenate([cat1_examples, cat2_examples], axis=0)
    return np.random.permutation(examples)


def generate_spurious_labels(
        majority_sp_label: int,
        minority_sp_label: int,
        class_size: int,
        minority_proportion: float) -> np.ndarray:
    """Generates spurious labels based on given spurious labels and proportions."""
    return np.random.choice([majority_sp_label, minority_sp_label],
                            size=(class_size,),
                            p=(1 - minority_proportion, minority_proportion))


def encode_context_x(token: np.ndarray) -> np.ndarray:
    token[:3] = np.array([1.0, 0.0, 0.0], dtype=token.dtype)
    return token


def encode_annotation(token: np.ndarray) -> np.ndarray:
    token[:3] = np.array([0.0, 1.0, 0.0], dtype=token.dtype)
    return token


def encode_query_x(token: np.ndarray) -> np.ndarray:
    token[:3] = np.array([0.0, 0.0, 1.0], dtype=token.dtype)
    return token


def simple_encode_x(token: np.ndarray, label: int = None, sp: int = None) -> np.ndarray:
    # encode class label if available
    if label is not None:
        token[0] = 2 * label - 1
    else:
        token[0] = 0

    # encode spurious feature if available
    if sp is not None:
        token[1] = 2 * sp - 1
    else:
        token[1] = 0

    # mark if this is a query
    if (label is None) and (sp is None):
        token[2] = 1
    else:
        token[2] = -1

    return token


def get_context_example_tokens(
        img_encoding: np.ndarray,
        x_spurious_token: np.ndarray,
        c_spurious_token: np.ndarray,
        class_token: np.ndarray,
        spurious_setting: str,
        simpler_construction: bool,
        label: int,
        sp: int,
) -> list[np.ndarray]:
    if not simpler_construction:
        if spurious_setting in ['inat_no_spurious', 'wb_erm', 'swap_erm']:
            return [encode_context_x(img_encoding), encode_annotation(class_token)]
        if spurious_setting == 'inat_sum_erm':
            return [encode_context_x(img_encoding + x_spurious_token), encode_annotation(class_token)]
        if spurious_setting in ['wb_dro', 'swap_dro']:
            return [encode_context_x(img_encoding), encode_annotation(class_token + c_spurious_token)]
        if spurious_setting == 'inat_sum_dro':
            return [encode_context_x(img_encoding + x_spurious_token), encode_annotation(class_token + c_spurious_token)]
        raise ValueError(f"Invalid spurious setting: '{spurious_setting}'.")
    else:
        if spurious_setting in ['inat_no_spurious', 'wb_erm']:
            return [simple_encode_x(img_encoding, label, sp=None)]
        if spurious_setting in ['wb_dro']:
            return [simple_encode_x(img_encoding, label, sp=sp)]
        raise ValueError(f"Invalid spurious setting: '{spurious_setting}'.")


def get_query_example_tokens(
        img_encoding: np.ndarray,
        x_spurious_token: np.ndarray,
        spurious_setting: str,
        simpler_construction: bool,
) -> list[np.ndarray]:
    if not simpler_construction:
        if spurious_setting in ['wb_erm', 'wb_dro', 'inat_no_spurious', 'swap_erm', 'swap_dro']:
            return [encode_query_x(img_encoding)]
        if spurious_setting in ['inat_sum_erm', 'inat_sum_dro']:
            return [encode_query_x(img_encoding + x_spurious_token)]
        raise ValueError(f"Invalid spurious setting: '{spurious_setting}'.")
    else:
        if spurious_setting in ['wb_erm', 'wb_dro', 'inat_no_spurious']:
            return [simple_encode_x(img_encoding, label=None, sp=None)]
        raise ValueError(f"Invalid spurious setting: '{spurious_setting}'.")


def get_group_counts_based_on_proportions(
        num_examples: int,
        group_proportions: list[float]) -> list[int]:
    """Computes group sizes based on proportions."""
    assert np.allclose(np.sum(group_proportions), 1.0)
    group_counts = [int(num_examples * p) for p in group_proportions]
    cur_sum = sum(group_counts)
    while cur_sum < num_examples:
        group_idx = np.random.choice(np.arange(4), p=group_proportions)
        group_counts[group_idx] += 1
        cur_sum += 1
    return group_counts
