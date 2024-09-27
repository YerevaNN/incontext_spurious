import numpy as np


class MultiSumSpurious:
    """
    This class adds spurious correlations to a dataset by modifying input image encodings through 
    repeated summation of spurious vectors.
    """

    def __init__(self, max_n_spurious, maximum_minority_prop):
        """
        Initializes the MultiSumSpurious class with the maximum number of spurious correlations
        and the maximum proportion of minority examples to be altered.

        Args:
            max_n_spurious (int): The upper limit on the number of spurious correlations.
            maximum_minority_prop (float): The maximum proportion of the minority class that can be modified.
        """
        self._max_n_spurious = max_n_spurious
        self._max_minority_prop = maximum_minority_prop

    def _generate_random_vector(self, norm, dim):
        """
        Generates a random vector with a given norm and dimension.

        Args:
            norm (float): The norm of the vector.
            dim (int): The dimensionality of the vector.

        Returns:
            np.ndarray: A random vector scaled to the desired norm.
        """
        random_vector = np.random.randn(dim)
        random_vector = (random_vector / np.linalg.norm(random_vector)) * norm
        return random_vector
    
    def _add_spurious_via_summation(self, labels, n_examples, max_minority_count, base_vector, queries=False):
        """
        Adds spurious correlations by modifying the input vectors through summation.

        Args:
            labels (np.ndarray): Array of original labels (0 or 1).
            n_examples (int): Number of examples in the instance.
            max_minority_count (int): Maximum number of minority examples to be modified.
            base_vector (np.ndarray): Base vector used for spurious modification.
            queries (bool, optional): Whether the current operation is for queries (True) or context (False).

        Returns:
            np.ndarray: Modified spurious vectors added to the dataset.
        """
        if not queries:
            minority_count = np.random.randint(low=1, high=max_minority_count)
        else:
            minority_count = int(n_examples / 4)  # Fixed proportion for queries

        spurious_scale = np.random.uniform(size=n_examples, low=0.0, high=1.0)
        random_spurious_vectors = spurious_scale.reshape(-1, 1) * base_vector.reshape(1, -1)

        spurious_labels = np.copy(labels)

        first_class_minority = np.random.choice(np.where(labels == 0)[0], size=minority_count, replace=False)
        sec_class_minority = np.random.choice(np.where(labels == 1)[0], size=minority_count, replace=False)

        spurious_labels[first_class_minority] = 1
        spurious_labels[sec_class_minority] = 0

        return (2 * spurious_labels - 1).reshape(-1, 1) * random_spurious_vectors

    def __call__(self, image_enc, context_labels, context_spurs, query_img_encodings, query_labels, query_spurs):
        """
        Applies spurious correlations to both the context and query data.

        Args:
            image_enc (np.ndarray): Encoded representations of the context images.
            context_labels (np.ndarray): Labels for the context images.
            context_spurs (np.ndarray): Spurious labels for the context images.
            query_img_encodings (np.ndarray): Encoded representations of the query images.
            query_labels (np.ndarray): Labels for the query images.
            query_spurs (np.ndarray): Spurious labels for the query images.

        Returns:
            tuple: The modified image encodings and spurious labels for both context and query datasets.
        """
        assert (context_labels == context_spurs).all()
        assert (context_labels == 0).sum() == (context_labels == 1).sum()
        assert (query_labels == query_spurs).all()

        mean_norm = np.linalg.norm(image_enc, axis=1).mean()
        count_of_spurious = np.random.randint(self._max_n_spurious)

        n_examples, dim = image_enc.shape
        n_examples_per_class = n_examples / 2
        max_minority_count = int(self._max_minority_prop * n_examples_per_class)

        for _ in range(count_of_spurious):
            base_vector = self._generate_random_vector(norm=mean_norm, dim=dim)

            image_enc += self._add_spurious_via_summation(
                labels=context_labels,
                n_examples=n_examples,
                max_minority_count=max_minority_count,
                base_vector=base_vector,
                queries=False
            )

            query_img_encodings += self._add_spurious_via_summation(
                labels=query_labels,
                n_examples=n_examples,
                max_minority_count=None,
                base_vector=base_vector,
                queries=True
            )

        return image_enc, context_spurs, query_img_encodings, query_spurs
