import numpy as np


class TokenGenerator:
    def __init__(self,
                 tokens_data,
                 are_x_spurious_tokens_fixed: bool,
                 are_c_spurious_tokens_fixed: bool,
                 are_class_tokens_fixed: bool,
                 token_generation_mode: str):
        """
        Initialize the TokenGenerator with token data and configuration flags.

        Parameters:
        tokens_data (dict): A dictionary containing token-related data such as 'token_len',
                            'avg_norm', 'fixed_spurious_tokens', and 'fixed_class_tokens'.
        are_x_spurious_tokens_fixed (bool): Flag indicating whether to use fixed spurious tokens
                                            when encoding spurious features to add to representations.
        are_c_spurious_tokens_fixed (bool): Flag indicating whether to use fixed spurious tokens
                                            when encoding spurious features alone.
        are_class_tokens_fixed (bool): Flag indicating whether to use fixed class tokens.
        token_generation_mode (str): Mode of token generation. Accepts 'random' or 'opposite'.
                                     'random' generates tokens with normal distribution, and 'opposite'
                                     generates a pair of tokens where the second is the negative of the first.
        """
        self.tokens_data = tokens_data
        self.are_x_spurious_tokens_fixed = are_x_spurious_tokens_fixed
        self.are_c_spurious_tokens_fixed = are_c_spurious_tokens_fixed
        self.are_class_tokens_fixed = are_class_tokens_fixed
        self.token_generation_mode = token_generation_mode

    def __call__(self):
        """
        Creates and returns token generators based on the configuration.

        When an instance of TokenGenerator is called, it will return generators
        that yield either fixed or random tokens for spurious and class tokens, depending on the configuration.

        Returns:
        tuple: A tuple containing two generators. The first generates spurious tokens,
               and the second generates class tokens.
        """

        def random_tokens_generator():
            """
            A generator function that yields random tokens continuously.

            Generates tokens with a normal distribution and normalizes them based on 'avg_norm'.
            """
            while True:
                if self.token_generation_mode == "random":
                    tokens = np.random.randn(2, self.tokens_data["token_len"]).astype(np.float32)
                else:
                    first_row = np.random.randn(1, self.tokens_data["token_len"]).astype(np.float32)
                    tokens = np.vstack((first_row, -first_row))

                tokens = (tokens / np.linalg.norm(tokens, axis=1, keepdims=True)) * self.tokens_data["avg_norm"]
                yield tokens

        def fixed_x_spurious_tokens_generator():
            while True:
                yield self.tokens_data[f"{self.token_generation_mode}_spurious_tokens"]

        def fixed_c_spurious_tokens_generator():
            while True:
                yield self.tokens_data[f"{self.token_generation_mode}_spurious_tokens_2"]

        def fixed_class_tokens_generator():
            while True:
                yield self.tokens_data[f"{self.token_generation_mode}_class_tokens"]

        x_spurious_tokens_generator = (fixed_x_spurious_tokens_generator
                                       if self.are_x_spurious_tokens_fixed
                                       else random_tokens_generator)()

        c_spurious_tokens_generator = (fixed_c_spurious_tokens_generator
                                       if self.are_c_spurious_tokens_fixed
                                       else random_tokens_generator)()

        class_tokens_generator = (fixed_class_tokens_generator
                                  if self.are_class_tokens_fixed
                                  else random_tokens_generator)()

        return x_spurious_tokens_generator, c_spurious_tokens_generator, class_tokens_generator
