from typing import List
import pandas as pd
from tqdm import tqdm
import re
import os

from torch.utils.data import DataLoader

from models import GeminiWriter
from dataset import ImageCaptionDataset, ImageCaptionCollate
from utils.utils import set_random_seed, read_prompt
from utils.data_processing import QuestionParser

import google.generativeai as genai

API_KEY = os.environ['API_KEY']  # Ensure API key is properly managed (e.g., via environment variables)


def configure_llm(config):
    """Set up and return the LLM with the provided configuration."""
    genai.configure(api_key=API_KEY)
    generation_config = genai.GenerationConfig(top_k=config['top_k'])
    system_prompt = read_prompt(config['system_prompt_path'])
    model = genai.GenerativeModel(
        model_name=config['model_ckpt'], 
        system_instruction=[system_prompt]
    )
    return GeminiWriter(config, model, generation_config=generation_config)

def contains_strings(lst):
    for element in lst:
        if isinstance(element, str):
            return True
        elif isinstance(element, list):
            if contains_strings(element):
                return True
    return False

def remove_rational(desc: str):
    try:
        if "New Description:" in desc:
            desc = desc.split('New Description:')[1].strip()
        elif "Description:" in desc:
            desc = desc.split('Description:')[1].strip()
        
        if "Rational" in desc:
            desc = desc.split('Rational')[0].strip(':*\n')
        if "Questions:" in desc:
            desc = desc.split('Questions:')[0].strip(':*\n')
        return desc
    except:
        return desc


def rewrite_captions(config, output_path):
    """Generate questions based on captions using the LLM."""
    set_random_seed(config["seed"])

    # Load dataset
    dataset = ImageCaptionDataset(config, df_path=os.path.join(output_path, 'difficult_questions_list.csv'), load_images=False)
    dataloader = DataLoader(
        dataset,
        collate_fn=ImageCaptionCollate,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    # Configure the LLM
    llm_sampler = configure_llm(config)

    writer = dfSaver(out_col_name="rewritten_caption", config=config, output_path=output_path)
    outputs = []

    for batch_idx, (_, captions, questions, _, _) in enumerate(tqdm(dataloader, desc="Rewriting Captions")):
        try:
            if contains_strings(questions):
                cur_outputs = llm_sampler.sample_from_model(captions, questions)[0]
                rewritten_captions = [remove_rational(cur_outputs)]
            else:
                rewritten_captions = captions
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")

        outputs.extend(rewritten_captions)

        # logging
        if batch_idx % config['log_every'] == 0 and config['activate_logging']:
            writer.save_df(outputs)

    # Final save
    writer.save_df(outputs)


class dfSaver:
    def __init__(self, out_col_name: str, config: dict, output_path: str):
        self.config = config
        self.output_path = output_path
        self.answers_df = pd.read_csv(os.path.join(output_path, 'difficult_questions_list.csv'))
        self.out_col_name = out_col_name

    def save_df(self, outputs: List[str]):
        df = self.answers_df.iloc[:len(outputs)].copy()
        df[self.out_col_name] = outputs
        df.to_csv(os.path.join(self.output_path, 'rewritten_captions.csv'), index=False)