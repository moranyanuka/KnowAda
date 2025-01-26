from PIL import Image
import os
import pandas as pd
from ast import literal_eval
from torch.utils.data import Dataset
from datasets import load_dataset


class BaseDataset(Dataset):
    def __init__(self, config, df_path=None, load_images=False):
        """
        Base dataset class to handle shared functionality across datasets.
        """
        self.config = config
        self.load_images = load_images
        self.df_path = df_path
        self.questions = None
        self.answers = None
        self.vlm_answers = None

        # if df_path:
        self._load_data()

    def _load_data(self):
        """Load questions, answers, and image indices from a CSV file."""
        if self.df_path:
            df = pd.read_csv(self.df_path)
        else:
            df = pd.DataFrame()
        try:
            df['questions'] = df['questions'].apply(literal_eval)
        except:
            pass
        self.questions = df['questions'].tolist() if 'questions' in df.columns else None
        self.answers = df['answers'].tolist() if 'answers' in df.columns else None
        self.vlm_answers = df['vlm_answers'].tolist() if 'vlm_answers' in df.columns else None
        self.image_idx = df['image_idx'].tolist() if 'image_idx' in df.columns else None

    def __len__(self):
        return len(self.questions or self.answers or self.descriptions)


class ImageCaptionDataset(BaseDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        dataset = load_dataset(config['dataset'], cache_dir=config['cache_dir'], trust_remote_code=True)[config['split']]
        self.data = dataset
        self.descriptions = dataset['description']

    def __getitem__(self, idx):
        """Retrieve an item by index."""
        caption = self.descriptions[int(self.image_idx[idx])] if self.image_idx else self.descriptions[idx]
        question = self.questions[idx] if self.questions else None
        answer = self.answers[idx] if self.answers else None
        vlm_answer = self.vlm_answers[idx] if self.vlm_answers else None
        image = self._load_image(idx)

        return {
            'image': image,
            'caption': caption,
            'question': question,
            'answer': answer,
            'vlm_answer': vlm_answer,
        }

    def _load_image(self, idx):
        """Load an image by index if applicable."""
        if not self.load_images:
            return None

        try:
            image = self.data[int(self.image_idx[idx])]['image'] if self.image_idx else self.data[idx]['image']

            return image
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None


class DocciWithSyntheticCaptionsDataset(BaseDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.descriptions = pd.read_csv(config['dataset'])['descriptions'].tolist()


def ImageCaptionCollate(batch):
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    vlm_answers = [item['vlm_answer'] for item in batch]

    return images, captions, questions, answers, vlm_answers
