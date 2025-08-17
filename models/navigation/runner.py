import json
import argparse
import logging
from dataclasses import dataclass, asdict


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(file, data):
  with open(file, 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

def get_venus_agent():
    from models.navigation.ui_venus_navi_agent import VenusNaviAgent
    return VenusNaviAgent

def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


@dataclass
class ModelConfig:
    model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct" 
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.6
    max_tokens: int = 2048
    max_pixels: int = 12845056
    min_pixels: int = 3136
    max_model_len: int = 10000
    max_num_seqs: int = 5
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1

    def __str__(self):
        return f"ModelConfig({', '.join(f'{k}={v}' for k, v in asdict(self).items())})"
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/root/models/uivenus-7B')
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int,  default=1)
    parser.add_argument("--input_file", type=str, default='examples/trace/trace.json')
    parser.add_argument("--output_file", type=str, default='./saved_trace.json')
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_pixels", type=int, default=12845056)
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--max_model_len", type=int, default=128000)
    parser.add_argument("--max_num_seqs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=0)
    
    args = parser.parse_args()

    logger = setup_logger("UI-vernus")

    model_config = ModelConfig(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=args.max_tokens,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        temperature=args.temperature,
        n=args.n,
    )
    logger.info(f"{model_config}")

    data = read_json(args.input_file)
    
    try:
        VenusNaviAgent = get_venus_agent()
        venus_agent = VenusNaviAgent(model_config, logger, args.history_length)
        logger.info("VenusNaviAgent initialized successfully")
    except Exception as e:
        logger.error(f"VenusNaviAgent initialized failed: {e}")
        raise

    results = []
    for trace_index, trace in enumerate(data):
        for item in trace:
            task = item['task']
            image_path = item['image_path']
            action_json = venus_agent.step(task, image_path)
        history_record = venus_agent.export_history()
        venus_agent.reset()
        results.append(history_record)

    save_json(args.output_file, results)


if __name__ == "__main__":
    main() 