import logging
import sys
import os
from absl import logging as absl_logging

def setup_logging():
    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除已有的处理器，避免重复
    if root_logger.handlers:
        root_logger.handlers.clear()

    # 添加控制台处理器
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # 获取进程信息
    process_id = os.getpid()
    process_name = os.path.basename(sys.argv[0])
    
    # 修改日志格式，添加进程信息
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [PID:%(process)d] - [%(processName)s] - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)