import os


def is_main_process():
    return os.environ.get('LOCAL_RANK', 0) == 0


def setup_aim_logger(logger):
    """
    Setup the directory structure for the 'aim' logger.

    This function takes the 'aim' logger object, extracts its unique hash,
    and then renames the current working directory to include this hash.
    It is specifically designed to work with the 'aim' logger in the context of a training setup.

    Args:
        logger (Logger): An instantiated logger object, specifically the 'aim' logger.

    Notes:
        The function assumes that the current working directory ends with a timestamp
        in the format ${now:%Y-%m-%d_%H-%M-%S}. It renames the current directory by appending
        the 'aim' logger's hash to the parent directory's name and then changes the current
        working directory to this new directory.
    """

    if not is_main_process():
        return

    # Extract the unique hash associated with the 'aim' logger
    aim_hash = logger.experiment.hash

    # Get the current working directory
    current_directory = os.getcwd()

    # Determine the parent directory of the current working directory
    parent_directory = os.path.dirname(current_directory)

    # Construct a new directory name by appending the 'aim' logger's hash
    new_directory = os.path.join(parent_directory, aim_hash)

    # Rename the current directory to the new directory name
    os.rename(current_directory, new_directory)

    # Change the current working directory to this new directory
    os.chdir(new_directory)
