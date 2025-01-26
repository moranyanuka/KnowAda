from typing import List
import pandas as pd
from tqdm import tqdm
import re
import os

from torch.utils.data import DataLoader

from judge import GeminiJudge
from dataset import ImageCaptionDataset, ImageCaptionCollate
from utils.utils import set_random_seed
from utils.data_processing import QuestionParser

import google.generativeai as genai

API_KEY = os.environ['API_KEY']

def configure_llm(config):
    """Set up and return the LLM with the provided configuration."""
    genai.configure(api_key=API_KEY)
    generation_config = genai.GenerationConfig(top_k=config['top_k'])
    system_prompt = "You are a helpful assistant proficient in analyzing vision reasoning problems."
    model = genai.GenerativeModel(
        model_name=config['model_ckpt'], 
        system_instruction=[system_prompt]
    )
    return GeminiJudge(config, model, generation_config=generation_config)


def judge_questions(config, output_path):
    """Generate questions based on captions using the LLM."""
    set_random_seed(config["seed"])

    # Load dataset
    dataset = ImageCaptionDataset(config, df_path=os.path.join(output_path, 'answers.csv'), load_images=False)
    dataloader = DataLoader(
        dataset,
        collate_fn=ImageCaptionCollate,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    # Configure the LLM
    llm_sampler = configure_llm(config)

    writer = dfSaver(out_col_name="judgments", config=config, output_path=output_path)
    outputs = []

    # Process data in batches
    for batch_idx, (_, captions, questions, _, vlm_answers) in enumerate(tqdm(dataloader, desc="Generating Judgements")):
        try:
            inputs_dict = {'<question>' : questions,
                           '<model-answer>': vlm_answers,
                           '<description>': captions}

            batch_judgments = llm_sampler.get_judgments(inputs_dict)
            match = re.search(r'\d+', batch_judgments[0])
            batch_judgments = [int(match.group())] if match else [-1]
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            batch_judgments = [-1] * len(captions)

        outputs.extend(batch_judgments)

        # Periodic logging
        if batch_idx % config['log_every'] == 0 and config['activate_logging']:
            # log_outputs(outputs, config, file_name=f"judgments.csv")
            writer.save_df(outputs)

    # Final save
    writer.save_df(outputs)

    # Classify the unknown questions and write to csv
    df_out = QuestionParser(config["difficulty_threshold"],
                            os.path.join(output_path, 'judgments.csv')).build_difficult_df()
    df_out.to_csv(os.path.join(output_path, f'difficult_questions_list.csv'))



class dfSaver:
    def __init__(self, out_col_name: str, config: dict, output_path: str):
        self.config = config
        self.output_path = output_path
        self.answers_df = pd.read_csv(os.path.join(output_path, 'answers.csv'))
        self.out_col_name = out_col_name

    def save_df(self, outputs: List[str]):
        df = self.answers_df.iloc[:len(outputs)].copy()
        df[self.out_col_name] = outputs
        df.to_csv(os.path.join(self.output_path, 'judgments.csv'), index=False)