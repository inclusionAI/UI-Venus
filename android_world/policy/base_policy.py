from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from openai import OpenAI
import requests
import logging
class LLMServer:
    def __init__(self, base_url: str, server_type='vllm', min_pixels=1280*720, max_pixels=1280*720):
        self.server_type = server_type
        self.logger = logging.getLogger(__name__)
        self.base_url = base_url
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        if server_type == 'vllm':
            self.llm = OpenAI(
                base_url=base_url,
                api_key="empty",
            )
    
    def generate_text(self, messages: list):
        retry_times = 10
        prediction = None
        if self.server_type == 'vllm':
            while retry_times > 0:
                try:
                    response = self.llm.chat.completions.create(
                        model="model",
                        messages=messages,
                        max_tokens=256,
                        temperature=0,
                        extra_body={"repetition_penalty": 1.05,
                                    "stop_token_ids": [],
                                "mm_processor_kwargs": {'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
                                }
                    )
                    prediction = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    retry_times -= 1
                    self.logger.error(f"Error when fetching response from client, with error: {e}; try again {retry_times}.")
            return prediction
        elif self.server_type == 'hf':
            while retry_times > 0:
                try:
                    response = requests.post(self.base_url, json=messages)
                    if response.status_code != 200:
                        retry_times -= 1
                        self.logger.error(f"Error when fetching response from client, with error: {response.text}; try again {retry_times}.")
                    else:
                        prediction = response.text
                        break
                except Exception as e:
                    retry_times -= 1
                    self.logger.error(f"Error when fetching response from client, with error: {e}; try again {retry_times}.")
                    
        else:
            raise ValueError(f"Unsupported server type: {self.server_type}")
        return prediction