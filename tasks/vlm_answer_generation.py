from typing import List
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import ImageCaptionDataset, ImageCaptionCollate
from utils.utils import set_random_seed
from utils.model_loading import load_vlm

from models import GeneralVLMCaptioner


def generate_answers(config, output_path):
    """Generate answers using a VLM based on an image and a question."""
    set_random_seed(config["seed"])

    # Load dataset
    dataset = ImageCaptionDataset(config, df_path=os.path.join(output_path, 'questions.csv'), load_images=True)
    dataloader = DataLoader(
        dataset,
        collate_fn=ImageCaptionCollate,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    tokenizer, model = load_vlm(config)
    vlm_sampler = GeneralVLMCaptioner(config, tokenizer, model)
    
    outputs = []
    writer = dfSaver(out_col_name="vlm_answers", config=config, output_path=output_path)

    # Process data in batches
    for batch_idx, (images, _, questions, _, _) in enumerate(tqdm(dataloader, desc="Sampling VLM Answers")):
        try:
            batch_answers = vlm_sampler.get_answers(questions, images)
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            batch_answers = [None] * len(questions)

        outputs.extend(batch_answers)

        # Periodic logging
        if batch_idx % config['log_every'] == 0 and config['activate_logging']:
            writer.save_df(outputs)

    writer.save_df(outputs)


class dfSaver:
    def __init__(self, out_col_name: str, config: dict, output_path: str):
        self.config = config
        self.output_path = output_path
        self.questions_df = pd.read_csv(os.path.join(output_path, 'questions.csv'))
        self.out_col_name = out_col_name

    def _replicate_rows_constant(self, df: pd.DataFrame, times: int):
        repeated_index = np.repeat(df.index, times)
        return df.loc[repeated_index].reset_index(drop=True)

    def save_df(self, outputs: List[str]):
        df = (
            self._replicate_rows_constant(self.questions_df, self.config['num_answers_per_question'])
            if self.config['num_answers_per_question'] > 1
            else self.questions_df
        )
        df = df.iloc[:len(outputs)].copy()
        df[self.out_col_name] = outputs
        df.to_csv(os.path.join(self.output_path, 'answers.csv'), index=False)