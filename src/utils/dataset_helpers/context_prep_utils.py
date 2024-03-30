import numpy as np


def generate_spurious_labels(
        primary_label: int,
        secondary_label: int,
        class_size: int,
        minority_proportion: float) -> list[int]:
    """Generates spurious labels based on given labels and proportions.

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

    return spurious_tokens


def get_context_example_tokens(
        img_encoding: np.ndarray,
        spurious_token: np.ndarray,
        class_token: np.ndarray,
        spurious_setting: str,
) -> list[np.ndarray]:
    if spurious_setting == 'separate_token':
        return [img_encoding, spurious_token, class_token]
    elif spurious_setting == 'sum':
        return [img_encoding + spurious_token, class_token]
    elif spurious_setting == 'no_spurious':
        return [img_encoding, class_token]
    else:
        raise ValueError(
            f"Invalid spurious setting: '{spurious_setting}'. Expected 'separate_token', 'sum', or 'no_spurious'.")


def get_query_example_tokens(
        img_encoding: np.ndarray,
        spurious_token: np.ndarray,
        spurious_setting: str,
) -> list[np.ndarray]:
    if spurious_setting == 'separate_token':
        return [img_encoding, spurious_token]
    elif spurious_setting == 'sum':
        return [img_encoding + spurious_token]
    elif spurious_setting == 'no_spurious':
        return [img_encoding]
    else:
        raise ValueError(
            f"Invalid spurious setting: '{spurious_setting}'. Expected 'separate_token', 'sum', or 'no_spurious'.")