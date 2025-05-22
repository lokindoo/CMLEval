import logging

import requests
import torch
from groq import Groq
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..utils.prompts import EVALUATION_SYS_PROMPT_DICT


class BaseLLM:
    def __init__(self, name):
        self.name = name


class LocalLLM(BaseLLM):
    def __init__(
        self, full_name: str, cache_path: str, qa_type: str, quant: str = "int8"
    ):
        """
        quant is int8 by default, but can be removed or changed
        """
        super().__init__(full_name.split("/")[-1])
        self.tokenizer = AutoTokenizer.from_pretrained(
            full_name,
            cache_dir=cache_path,
            use_fast=True,
        )

        self.sys_prompt = EVALUATION_SYS_PROMPT_DICT[qa_type]

        model_kwargs = {
            "cache_dir": cache_path,
            "device_map": "auto",
            "local_files_only": True,
        }

        if quant == "int8":
            model_kwargs["load_in_8bit"] = True

        elif quant == "int4":
            qb_conf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = qb_conf

        self.model = AutoModelForCausalLM.from_pretrained(
            full_name,
            **model_kwargs,
        )

    def predict(self, prompt):
        inputs = self.tokenizer(
            self.sys_prompt + "\n" + prompt, return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=2000)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class OpenAILLM(BaseLLM):
    def __init__(self, name: str, api_key: str, qa_type: str):
        super().__init__(name)
        self.model = name
        self.client = OpenAI(api_key=api_key)
        self.sys_prompt = EVALUATION_SYS_PROMPT_DICT[qa_type]

    def predict(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content


class GroqLLM(BaseLLM):
    def __init__(self, name: str, api_key: str, qa_type: str):
        super().__init__(name)
        self.model = name
        self.client = Groq(api_key=api_key)
        self.sys_prompt = EVALUATION_SYS_PROMPT_DICT[qa_type]

    def predict(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content


class RemoteLLM(BaseLLM):
    def __init__(self, name: str, endpoint: str, qa_type: str, api_key=None):
        super().__init__(name)
        self.endpoint = endpoint
        self.sys_prompt = EVALUATION_SYS_PROMPT_DICT[qa_type]
        self.api_key = api_key

    def predict(self, prompt):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"prompt": self.sys_prompt + "\n" + prompt}
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json().get("response", "")
        except Exception as e:
            logging.error(f"Error calling {self.name} at {self.endpoint}: {e}")
            result = ""
        return result


company2wrapper = {
    "DEFAULT": RemoteLLM,
    "OPENAI": OpenAILLM,
    "GROQ": GroqLLM,
}
