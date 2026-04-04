import os
import json
import time
import uuid
from typing import List, Dict, Any, Union, Optional, Tuple
from multiprocessing import Pool, Process, Value
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'
)


RESP_DELIMITER = '====LLM4JMH===='

class BaseModel:
    max_tokens = 4096

    model_name = None
    model = None

    def set_dynamic_config(self):
        self.temperature = 0.2
        self.top_p = None
        self.top_k = None

    def set_deterministic_config(self):
        self.temperature = 0.0
        self.top_k = None
        self.top_p = None

    def generate(self, message: str, *, deterministic: bool = True) -> Tuple[bool, str]:
        if deterministic:
            self.set_deterministic_config()
        else:
            self.set_dynamic_config()

        try:
            logging.info(f"[{self.model_name}-generate] temperature: {self.temperature}")
            ret = self._generate(message)
            return True, ret
        except Exception as ex:
            logging.error(f"An error occurred while querying the {self.model_name}: {ex}")
            err_msg = str(ex)
        return False, err_msg

    def generate_multi_outputs(self, message: str, *, num_outputs: int = 1) -> List[Tuple[bool, str]]:
        # NOTE: deepseek already generate the same response
        def task(_):
            return self.generate(message, deterministic=False)

        with ThreadPoolExecutor(num_outputs) as executor:
            return list(executor.map(task, range(num_outputs)))

    def _generate(self, message: str, examples: List[Any] = []):
        raise NotImplementedError("Derived class must implement function `_generate`")

    def generate_with_examples(self, message: str, examples: List[Any], *, deterministic: bool = True) -> Tuple[bool, str]:
        if deterministic:
            self.set_deterministic_config()
        else:
            self.set_dynamic_config()

        if len(examples) % 2 != 0:
            raise Exception(f"The number of given examples must be even, now it's {len(examples)}")

        try:
            logging.info(f"[{self.model_name}-generate_with_examples] temperature: {self.temperature}, examples: {len(examples)}")
            ret = self._generate(message, examples)
            return True, ret
        except Exception as ex:
            logging.error(f"An error occurred while querying the {self.model_name}: {ex}")
            err_msg = str(ex)
        return False, err_msg

    def generate_multi_outputs_with_examples(self, message: str, examples: List[Any], *, num_outputs: int = 1) -> List[Tuple[bool, str]]:
        def task(_):
            return self.generate_with_examples(message, examples, deterministic=False)

        with ThreadPoolExecutor(num_outputs) as executor:
            return list(executor.map(task, range(num_outputs)))


class DeepSeek(BaseModel):
    def __init__(self, *, model_name: str = 'deepseek-chat', system_prompt: str = None):
        from openai import OpenAI
        self.model_name = model_name
        self.model = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com', timeout=120)
        self.system_prompt = system_prompt

    def _generate(self, message: str, examples: List[Any] = []):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        for i in range(0, len(examples), 2):
            messages.append({"role": "user", "content": examples[i]})
            messages.append({"role": "assistant", "content": examples[i+1]})

        messages.append({"role": "user", "content": message})
        logging.info("==================================================")
        print(messages)
        logging.info("==================================================")
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # deterministic or dynamic responses
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=1,
        )
        raw_message = response.choices[0].message
        if raw_message.reasoning_content:
            return f'Thinking: {raw_message.reasoning_content}\n{RESP_DELIMITER}\n{raw_message.content}'
        else:
            return raw_message.content


class Gemini(BaseModel):
    def __init__(self, *, model_name: str = 'gemini-2.0-flash', system_prompt: str = None):
        self.system_prompt = system_prompt
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name) if self.system_prompt is None else genai.GenerativeModel(self.model_name, system_instruction=self.system_prompt)

    def _generate(self, message: str, examples: List[Any] = []):
        messages = []
        for i in range(0, len(examples), 2):
            messages.append({"role": "user", "parts": [{"text": examples[i]}]})
            messages.append({"role": "model", "parts": [{"text": examples[i+1]}]})

        messages.append({"role": "user", "parts": [{"text": message}]})
        response = self.model.generate_content(messages, generation_config={"temperature": self.temperature, "top_p": self.top_p, "max_output_tokens": self.max_tokens, "candidate_count": 1}, request_options={"timeout": 120})
        return response.text


class Gpt(BaseModel):
    def __init__(self, *, model_name: str = 'gpt-3.5-turbo', system_prompt: str = None):
        from openai import OpenAI
        self.model_name = model_name
        self.model = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'), timeout=120)
        self.system_prompt = system_prompt

    def _generate(self, message: str, examples: List[Any] = []):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        for i in range(0, len(examples), 2):
            messages.append({"role": "user", "content": examples[i]})
            messages.append({"role": "assistant", "content": examples[i+1]})

        messages.append({"role": "user", "content": message})
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # deterministic or dynamic responses
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=1,
        )
        return response.choices[0].message.content


class OllamaModel(BaseModel):
    def __init__(self, *, model_name: str, system_prompt: str = None, **kwargs):
        from openai import OpenAI
        self.model_name = model_name
        self.model = OpenAI(api_key="ollama", base_url='http://ollama.fokus.fraunhofer.de:11434/v1', timeout=1000)
        self.system_prompt = system_prompt

    def _generate(self, message: str, examples: List[Any] = []):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        for i in range(0, len(examples), 2):
            messages.append({"role": "user", "content": examples[i]})
            messages.append({"role": "assistant", "content": examples[i+1]})

        messages.append({"role": "user", "content": message})
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            # deterministic or dynamic responses
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=1,
        )
        return response.choices[0].message.content


class OllamaDeepSeekV3(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        # super().__init__(model_name="deepseek-v3:latest", system_prompt=system_prompt)
        # super().__init__(model_name="deepseek-v3-671b-fixed:latest", system_prompt=system_prompt)
        super().__init__(model_name="deepseek-v3.1:latest", system_prompt=system_prompt)


class OllamaQwen3Coder480B(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        super().__init__(model_name="qwen3-coder:480b", system_prompt=system_prompt)


class OllamaQwen3Coder30B(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        super().__init__(model_name="qwen3-coder:30b", system_prompt=system_prompt)


class OllamaQwen25Coder32B(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        super().__init__(model_name="qwen2.5-coder:32b", system_prompt=system_prompt)

class OllamaQwen25Coder7B(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        super().__init__(model_name="qwen2.5-coder:7b", system_prompt=system_prompt)

class OllamaGptOss120B(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        super().__init__(model_name="gpt-oss:120b", system_prompt=system_prompt)


class OllamaQwen3_8B(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        super().__init__(model_name="qwen3:8b", system_prompt=system_prompt)


class OllamaQwen3_32B(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        super().__init__(model_name="qwen3:32b", system_prompt=system_prompt)

class OllamaQwen3_17B(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        super().__init__(model_name="qwen3:1.7b", system_prompt=system_prompt)

class OllamaLlama31_8B(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        super().__init__(model_name="llama3.1:8b", system_prompt=system_prompt)

class OllamaGemma2_9B(OllamaModel):
    def __init__(self, *, system_prompt: str = None, **kwargs):
        super().__init__(model_name="gemma2:9b", system_prompt=system_prompt)


model_map = {
    "gemini-2.0-flash": Gemini,
    "deepseek-chat": DeepSeek,
    "deepseek-reasoner": DeepSeek,
    'gpt-3.5-turbo': Gpt,
    'ollama-deepseek-v3': OllamaDeepSeekV3,
    # 'ollama-qwen3-coder:480b': OllamaQwen3Coder480B,
    # 'ollama-qwen3-coder-30b': OllamaQwen3Coder30B,
    'ollama-qwen2.5-coder-32b': OllamaQwen25Coder32B,
    'ollama-qwen2.5-coder-7b': OllamaQwen25Coder7B,
    'ollama-gptoss-120b': OllamaGptOss120B,

    'ollama-qwen3-1.7b': OllamaQwen3_17B,
    'ollama-qwen3-8b': OllamaQwen3_8B,
    'ollama-qwen32-8b': OllamaQwen3_32B,
    'ollama-llama3.1-8b': OllamaLlama31_8B,
    'ollama-gemma2-9b': OllamaGemma2_9B,
}

def generate(model_name: str, prompt: str, *, system_prompt: str = None, num_outputs: int = 1, multi_output_mode: bool = False):

    if model_name not in model_map:
        raise ValueError(f"Not support model {model_name}")

    client = model_map[model_name](model_name=model_name, system_prompt=system_prompt)
    if num_outputs > 1 or multi_output_mode is True:
        return client.generate_multi_outputs(prompt, num_outputs=num_outputs)
    return client.generate(prompt)


def generate_with_examples(model_name: str, message: str, examples: List[str] | str, *, system_prompt: str = None, num_outputs: int = 1, multi_output_mode: bool = False):
    if model_name not in model_map:
        raise ValueError(f"Not support model {model_name}")

    client = model_map[model_name](model_name=model_name, system_prompt=system_prompt)
    if isinstance(examples, list):
        if len(examples) % 2 != 0:
            raise Exception("`examples` is not configured correctly, the length must be `2n+1`")

    if num_outputs > 1 or multi_output_mode is True:
        return client.generate_multi_outputs_with_examples(message, examples=examples, num_outputs=num_outputs)

    return client.generate_with_examples(message, examples=examples)


if __name__ == "__main__":
    num_outputs = 3
    text_prompt = "Sentence: My cat just knocked over a glass of water. Mood: "
    # model_names = ['deepseek-chat', 'gemini-2.0-flash']
    # 'ollama-qwen3-coder-30b', 'ollama-qwen3-coder-480b'
    # model_names = ['ollama-deepseek-v3']
    model_names = ['ollama-qwen2.5-coder-32b']
    for model_name in model_names:
        response = generate(model_name, text_prompt, num_outputs=1)
        print(f"{model_name}'s response:\n{response}")
        print("---------------------------------------------------")
        responses = generate(model_name, text_prompt, num_outputs=num_outputs)
        for idx, response in enumerate(responses):
            print(f"{model_name}-{idx}'s response:\n{response}")

        print("===================================================")

    examples = [
        "Sentence: The sun is shining and the birds are singing. Mood: ",
        "Happy",
        "Sentence: I missed my bus and now I'm going to be late. Mood: ",
        "Sad",
    ]
    for model_name in model_names:
        response = generate_with_examples(model_name, text_prompt, examples, num_outputs=1)
        print(f"{model_name}'s response:\n{response}")
        print("---------------------------------------------------")
        responses = generate_with_examples(model_name, text_prompt, examples, num_outputs=num_outputs)
        for idx, response in enumerate(responses):
            print(f"{model_name}-{idx}'s response:\n{response}")

        print("===================================================")
