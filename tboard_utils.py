from tensorboardX import SummaryWriter


# A `SummaryWriter` instance initialized via `init_tensorboard()` and acquired
# with `get_tensorboard()`. This paradigm allows the writer to be acquired
# from anywhere (e.g., within a model or optimizer) without the writer having
# to be explicitly passed as a parameter to every object.
_TENSORBOARD_WRITER = None


def init_tensorboard(logdir: str, **kwargs):
    """Create a Tensorboard SummaryWriter instance that writes to `logdir`.

    The writer is saved as a module-level object. It can then be acquired and
    used in any file by calling `get_tensorboard()`.

    Args:
        logdir (str):
            A directory path to which the created SummaryWriter should output
            event files.
        kwargs (Dict[str, Any]):
            Any additional keyword arguments that should be supplied to the
            new SummaryWriter.
    """
    global _TENSORBOARD_WRITER
    _TENSORBOARD_WRITER = SummaryWriter(logdir, **kwargs)


def get_tensorboard() -> SummaryWriter:
    """
    Acquire the Tensorboard SummaryWriter instance created with
    `init_tensorboard()`.

    Returns:
        (SummaryWriter):
            The SummaryWriter instance created by the most recent call to
            `init_tensorboard()`.
    """
    global _TENSORBOARD_WRITER
    assert _TENSORBOARD_WRITER is not None, (
        "get_tensorboard() called before init_tensorboard(); please specify "
        "a logdir to init_tensorboard() first."
    )
    return _TENSORBOARD_WRITER