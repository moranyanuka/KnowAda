import time
from copy import deepcopy
import functools

from utils.utils import read_prompt
from models import GeminiBase


class GeminiJudge(GeminiBase):
    """
    A class for generating judgments using Gemini.
    This class is responsible for formatting prompts and interacting with the model
    to generate responses based on input descriptions and questions.
    """
    def __init__(self, config, model, generation_config=None):
        """
        Initialize the GeminiJudge class.

        Args:
            config (dict): Configuration dictionary containing settings.
            model (Model): The LLM model instance.
            generation_config (optional): Additional generation configuration.
        """
        super().__init__(config, model=model, generation_config=generation_config)
        self.prompt = self._create_initial_prompt(config)

    def _create_initial_prompt(self, config):
        """
        Load the initial judgment prompt from the configuration.

        Args:
            config (dict): Configuration dictionary containing the prompt path.

        Returns:
            str: The loaded judgment prompt.
        """
        return read_prompt(config['prompt_path'])

    def _create_judge_prompt(self, inputs_dict):
        """
        Create a customized judgment prompt by replacing placeholders with input values.

        Args:
            inputs_dict (dict): Dictionary containing input keys and their corresponding values.

        Returns:
            str: The formatted judgment prompt.
        """
        prompt = self.prompt
        for key, value in inputs_dict.items():
            if not value or value[0] is None:
                continue
            try:
                prompt = prompt.replace(key, value[0])
            except KeyError:
                print(f"Key {key} not found in inputs_dict during prompt creation.")
                prompt = prompt.replace(key, "")
        return prompt

    def _format_prompt(self, prompt):
        """
        Format the prompt to be compatible with the model's input requirements.

        Args:
            prompt (str): The raw prompt string.

        Returns:
            list[dict]: The formatted prompt as a list of dictionaries.
        """
        return [{"role": "user", "parts": prompt}]

    @functools.lru_cache(maxsize=100)
    def _get_answer(self, prompt):
        """
        Get an answer from the model given a prompt.

        This method uses caching to avoid redundant requests for the same prompt.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The model's response or an error message in case of failure.
        """
        formatted_prompt = self._format_prompt(prompt)[0]
        try:
            chat = self.model.start_chat(history=formatted_prompt)
            response = self.api_req(chat, formatted_prompt)
        except Exception as e:
            # Retry after a delay in case of failure
            time.sleep(120)
            try:
                chat = self.model.start_chat(history=formatted_prompt)
                response = self.api_req(chat, formatted_prompt)
            except Exception as retry_exception:
                return f"Failed to get response: {retry_exception}"
        return response.text.strip()

    def get_judgments(self, inputs_dict):
        """
        Generate judgments based on the given inputs.

        Args:
            inputs_dict (dict): Dictionary containing the input details for the judgment.
            generate_gt_answer (bool, optional): Flag to indicate whether to generate
                                                 a ground truth answer. Default is False.

        Returns:
            tuple: A tuple containing the ground truth answer (if generated) and the model's judgment response.
        """
        prompt = self._create_judge_prompt(inputs_dict)
        response = self._get_answer(prompt)
        return [response]