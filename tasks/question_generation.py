from typing import List
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import GeminiWriter
from dataset import ImageCaptionDataset, ImageCaptionCollate
from utils.utils import read_prompt, set_random_seed
from utils.data_processing import save_df

import google.generativeai as genai

API_KEY = os.environ['API_KEY']


def configure_llm(config):
    """Set up and return the LLM with the provided configuration."""
    genai.configure(api_key=API_KEY)
    system_prompt = read_prompt(os.path.join(config['prompt_files_path'], 'system/system_prompt.txt'))
    generation_config = genai.GenerationConfig(top_k=config['top_k'])
    model = genai.GenerativeModel(
        model_name=config['model_ckpt'], 
        system_instruction=[system_prompt]
    )
    return GeminiWriter(config, model, generation_config=generation_config)


def generate_questions(config, output_path):
    """Generate questions based on captions using the LLM."""
    set_random_seed(config["seed"])

    # Load dataset
    dataset = ImageCaptionDataset(config, df_path=None)
    dataloader = DataLoader(
        dataset,
        collate_fn=ImageCaptionCollate,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    # Configure the LLM
    llm_sampler = configure_llm(config)

    outputs = []

    # Process data in batches
    for batch_idx, (_, captions, _, _, _) in enumerate(tqdm(dataloader, desc="Generating Questions")):
        try:
            batch_questions = llm_sampler.sample_from_model(captions, questions=[None])
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            batch_questions = ['What is the main subject of the image?'] * len(captions)

        outputs.extend(batch_questions)

        # Periodic logging
        if batch_idx % config['log_every'] == 0 and config['activate_logging']:
            log_outputs(outputs, file_name=f"questions.csv", output_path=output_path)

    # Final save
    log_outputs(outputs, file_name="questions.csv", output_path=output_path)


def log_outputs(outputs: List[str], file_name: str, output_path: str):
    """Helper function to save generated outputs to a file."""
    save_df(
        input_list=[outputs],
        column_list=['generated_questions'],
        file_name=file_name,
        process_data=True, 
        output_path=output_path
    )