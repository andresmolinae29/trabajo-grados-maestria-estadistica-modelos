

class ModelNotFitException(Exception):
    """Raised when the model is not fit yet."""
    pass


class DataLoaderException(Exception):
    """Raised when there is an error loading the data."""
    pass