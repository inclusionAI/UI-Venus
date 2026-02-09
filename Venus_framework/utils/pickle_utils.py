import gzip
import pickle
from typing import Any


def gzip_pickle(data: Any) -> bytes:
    """将数据序列化并压缩为 gzip 格式
    
    Args:
        data: 任意可 pickle 的数据
        
    Returns:
        压缩后的字节数据
    """
    pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    return gzip.compress(pickled)


def load_gzip_pickle(filepath: str) -> Any:
    """从 gzip 压缩的 pickle 文件加载数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        反序列化后的数据
    """
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)
