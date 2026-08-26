import gzip
import pickle
from typing import Any


def gzip_pickle(data: Any) -> bytes:
    """Serialize data and compress it with gzip.
    
    Args:
        data: Any pickle-compatible value.
        
    Returns:
        Compressed bytes.
    """
    pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    return gzip.compress(pickled)


def load_gzip_pickle(filepath: str) -> Any:
    """Load data from a gzip-compressed pickle file.
    
    Args:
        filepath: File path.
        
    Returns:
        Deserialized data.
    """
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)
