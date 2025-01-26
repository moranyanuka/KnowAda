import os
import time
import logging
from typing import List, Dict, Optional

import torch
from utils.utils import read_prompt
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# Constants
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}


class BaseModel:
    """Base class for vision-language models."""

    def __init__(self, config: Dict, model: object):
        """
        Initialize the base model.

        Args:
            config (Dict): Configuration dictionary.
            model (object): Pre-initialized model object.
        """
        self.config = config
        self.model = model

    def sample_from_model(self, image_descriptions: List[str]):
        """Raise an error if called directly from the base class."""
        raise NotImplementedError("This method should not be called from the base class.")


class VLMModel(BaseModel):
    """Vision-Language Model wrapper with processor integration."""

    def __init__(self, config: Dict, processor: object, model: object):
        super().__init__(config, model)
        self.device = config.get("device", "cpu")
        self.processor = processor


class TinyLLavaVLMACaptioner(VLMModel):
    """Assistant for TinyLLaVA Vision-Language Model."""

    def __init__(self, config: Dict, processor: object, model: object):
        super().__init__(config, processor, model)
        self.prompt = read_prompt(config["prompt_path"])

    def get_answers(self, questions: Optional[List[str]] = None, images: Optional[List[object]] = None) -> List[str]:
        """
        Generate answers using the model.

        Args:
            questions (Optional[List[str]]): List of questions.
            images (Optional[List[object]]): List of image objects.

        Returns:
            List[str]: Generated answers.
        """
        prompts = [self.prompt.replace("<question>", q) for q in questions] if questions else [self.prompt] * len(images)
        do_sample = self.num_answers_per_question > 1
        temperature = self.config.get("sampling_temperature") if do_sample else None

        with torch.inference_mode():
            output_text, generation_time = self.model.chat(
                prompt=prompts[0],
                image=images[0],
                tokenizer=self.processor,
                temperature=temperature,
                max_new_tokens=self.config.get("max_new_tokens", 100),
                num_return_sequences=self.num_answers_per_question,
            )
        return output_text


class GeneralVLMCaptioner(VLMModel):
    """General Hugging Face Vision-Language Model Assistant."""

    def __init__(self, config: Dict, processor: object, model: object):
        super().__init__(config, processor, model)
        self.prompt = read_prompt(config["prompt_path"])
        self.num_answers_per_question = config.get("num_answers_per_question", 10)

    def get_answers(self, questions: Optional[List[str]] = None, images: Optional[List[object]] = None) -> List[str]:
        """
        Generate answers using the model.

        Args:
            questions (Optional[List[str]]): List of questions.
            images (Optional[List[object]]): List of image objects.

        Returns:
            List[str]: Generated answers.
        """
        prompts = [self.prompt.replace("<question>", q) for q in questions] if questions else [self.prompt] * len(images)
        do_sample = self.num_answers_per_question > 1
        temperature = self.config.get("sampling_temperature") if do_sample else None

        inputs = self.processor(prompts, images, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_ids_size = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 100),
                do_sample=do_sample,
                temperature=temperature,
                num_return_sequences=self.num_answers_per_question,
            )
        return self.processor.batch_decode(output_ids[:, input_ids_size:], skip_special_tokens=True)


class GeminiBase(BaseModel):
    """Base class for Gemini model-based assistants."""

    def __init__(self, config: Dict, model: object, generation_config: Optional[object] = None):
        super().__init__(config, model)
        self.generation_config = generation_config

    def api_req(self, chat, prompt: List[Dict]) -> object:
        """
        Make an API request.

        Args:
            chat: Chat object.
            prompt (List[Dict]): Chat prompt.

        Returns:
            object: API response.
        """
        return chat.send_message(content=prompt, generation_config=self.generation_config, safety_settings=SAFETY_SETTINGS)


class GeminiWriter(GeminiBase):
    """Gemini-based question generation assistant."""

    def __init__(self, config: Dict, model: object, generation_config: Optional[object] = None):
        super().__init__(config, model, generation_config)
        self.prompt = self.create_initial_prompt_detailed_description(config)

    @staticmethod
    def create_initial_prompt_detailed_description(config: Dict) -> List[Dict]:
        """
        Create a detailed initial prompt.

        Args:
            config (Dict): Configuration dictionary.

        Returns:
            List[Dict]: Detailed prompts.
        """
        prompt_files = sorted(
            os.path.join(config["prompt_files_path"], f) for f in os.listdir(config["prompt_files_path"]) if f.endswith(".txt")
        )
        prompts = []
        for i in range(0, len(prompt_files), 2):
            prompts.append({"role": "user", "parts": read_prompt(prompt_files[i])})
            prompts.append({"role": "model", "parts": read_prompt(prompt_files[i + 1])})
        return prompts

    def sample_from_model(self, image_descriptions: List[str], questions: Optional[List[str]] = None) -> List[str]:
        """
        Sample from the model.

        Args:
            image_descriptions (List[str]): List of image descriptions.
            questions (Optional[List[str]]): List of questions.

        Returns:
            List[str]: Generated responses.
        """
        prompt = self.prompt
        chat = self.model.start_chat(history=prompt)
        prompt = {"role": "user", "parts": f"{image_descriptions[0]}" if not questions else f"{image_descriptions[0]}\n\n{questions[0]}"}

        try:
            response = self.api_req(chat, prompt)
        except Exception as e:
            logging.error(f"Error in API request: {e}. Retrying...")
            time.sleep(120)
            chat = self.model.start_chat(history=prompt)
            response = self.api_req(chat, prompt)

        return [response.text.strip()]