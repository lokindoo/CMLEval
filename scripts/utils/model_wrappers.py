import logging

import requests
from groq import Groq
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.prompts import EVALUATION_SYS_PROMPT


class BaseLLM:
    def __init__(self, name):
        self.name = name


class LocalLLM(BaseLLM):
    def __init__(self, full_name, cache_path):
        super().__init__(full_name.split("/")[-1])
        self.tokenizer = AutoTokenizer.from_pretrained(full_name, cache_dir=cache_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            full_name, cache_dir=cache_path
        )

    def predict(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=200)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result


class OpenAILLM(BaseLLM):
    def __init__(self, name: str, api_key: str):
        super().__init__(name)
        self.model = name
        self.client = OpenAI(api_key=api_key)

    def predict(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": EVALUATION_SYS_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content


class GroqLLM(BaseLLM):
    def __init__(self, name: str, api_key: str):
        super().__init__(name)
        self.model = name
        self.client = Groq(api_key=api_key)

    def predict(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": EVALUATION_SYS_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content


class RemoteLLM(BaseLLM):
    def __init__(self, name, endpoint, api_key=None):
        super().__init__(name)
        self.endpoint = endpoint
        self.api_key = api_key

    def predict(self, prompt):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"prompt": prompt}
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
