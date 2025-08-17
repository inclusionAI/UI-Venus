import os
from typing import Dict, List, Any, Tuple

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from qwen_vl_utils import process_vision_info


class NaviVLLM:
    def __init__(self, model_config, logger):
        """
        Initialize the NaviVLLM model.

        Args:
            model_config: Configuration object with model parameters.
            logger: Logger instance for logging.
        """
        self.logger = logger
        self.model_config = model_config

        self.model = LLM(
            model=model_config.model_path,
            max_model_len=model_config.max_model_len,
            max_num_seqs=model_config.max_num_seqs,
            tensor_parallel_size=model_config.tensor_parallel_size,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            )
        self.processor = AutoProcessor.from_pretrained(model_config.model_path)
        self.processor.image_processor.max_pixels = model_config.max_pixels
        self.processor.image_processor.min_pixels = model_config.min_pixels
        self.sampling_params = SamplingParams(
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            top_k=model_config.top_k,
            repetition_penalty=1.05,
            n=model_config.n,
            stop_token_ids=[]
        )

        self.logger.info(
            f"SamplingParams: max_tokens={model_config.max_tokens}, "
            f"temperature={model_config.temperature}, top_p={model_config.top_p}, "
            f"top_k={model_config.top_k}, n={model_config.n}, "
            f"stop_token_ids={self.sampling_params.stop_token_ids}"
        )

    def create_message_for_image(self, image: str, problem: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": problem},
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": self.model_config.min_pixels,
                        "max_pixels": self.model_config.max_pixels,
                    }
                ],
            },
        ]

    def _prepare_llm_inputs(self, messages_list: List[List[Dict]]) -> List[Dict]:
        """
        Convert messages to vLLM input format with multi-modal data.

        Args:
            messages_list: List of message lists (one per sample).

        Returns:
            List of dictionaries containing 'prompt' and 'multi_modal_data'.
        """
        prompts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
    
        image_inputs, _ = process_vision_info(messages_list)
        
        llm_inputs = []
        for image_input, text in zip(image_inputs, prompts):
            mm_data = {"image": image_input}
            llm_inputs.append({
                "prompt": text,
                "multi_modal_data": mm_data
            })
        
        return llm_inputs

    def _process_batch(
        self,
        batch_data: List[Tuple[bytes, str, str]], 
    ) -> Tuple[List[str], List[str]]:
        """
        Process a batch of (image_path, problem) data into vLLM inputs.

        Args:
            batch_data: List of tuples (image_path, problem).

        Returns:
            List of LLM input dictionaries.
        """
        images, problems = zip(*batch_data)
        
        messages_list = [self.create_message_for_image(img, prob) for img, prob in zip(images, problems)]
        
        return self._prepare_llm_inputs(messages_list)
        
        
    def __call__(self, data, print_log=False):
        """
        Generate responses for a list of (image_path, problem) pairs.

        Args:
            data: List of tuples (image_path, problem).
            print_log: Whether to log questions and answers.

        Returns:
            List[List[str]]: Each inner list contains `n` generated responses.
        """
        llm_input_list = self._process_batch(data)

        outputs = self.model.generate(llm_input_list, sampling_params=self.sampling_params)
        responses = []
        for output in outputs:
            generated_texts = [o.text for o in output.outputs]
            responses.append(generated_texts)

        if print_log:
            for (image_path, problem), response in zip(data, responses):
                self.logger.info(f"Image: {os.path.basename(image_path)}")
                self.logger.info(f"Problem: {problem}")
                self.logger.info(f"Response: {response[0]}")
                self.logger.info("-" * 50)

        return responses