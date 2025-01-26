import json
import os
import shutil
import torch
import numpy as np
import random
from typing import Dict
import datetime
import argparse

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--generate_questions", type=bool, default=True, help="Whether to generate questions based on image captions")
    parser.add_argument("--generate_answers", type=bool, default=True, help="Whether to sample answers with the VLM")
    parser.add_argument("--generate_judgments", type=bool, default=True, help="Whether to judge the generated answers")
    parser.add_argument("--generate_rewritten_descriptions", type=bool, default=True, help="Whether to rewrite the captions based on the judgments")
    parser.add_argument("--output_folder", type=str, default='./outputs', help="Where to save the outputs")
    args = parser.parse_args()
    validate_args(args)
    return args

def validate_args(args: argparse.Namespace):
    """Validate argument combinations."""
    if not args.generate_questions and args.generate_answers and not os.path.exists(os.path.join(args.output_folder, "questions.csv")):
        raise ValueError(
            "Error: To start from 'generate_answers', you need the intermediate questions file generated "
            "from 'generate_questions' "
        )
    if not args.generate_answers and args.generate_judgments and not os.path.exists(os.path.join(args.output_folder, "answers.csv")):
        raise ValueError(
            "Error: To start from 'judge', you need the intermediate judgments file generated "
            "from 'generate_answers'."
        )
    if not args.generate_judgments and args.generate_rewritten_descriptions and not os.path.exists(os.path.join(args.output_folder, "difficult_questions_list.csv")):
        raise ValueError(
            "Error: To start from 'rewrite', you need the intermediate file with difficult questions generated "
            "from 'judge'."
        )
    print("Arguments validation successful.")

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def add_datetime(config: Dict):
    dt = datetime.datetime.now()
    config['output_path'] = os.path.join(
        config['output_path'],
        f"{dt.strftime('%x').replace('/', '_')}___{dt.strftime('%X').replace(':', '_')}"
    )

def read_prompt(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def get_all_files(directory_path: str):
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files

def write_config_with_files(config: Dict):
    prompt_files = get_all_files(config["prompt_files_path"])
    path = config["output_path"]
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "generation_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    for prompt_file in prompt_files:
        shutil.copyfile(prompt_file, os.path.join(config["output_path"], os.path.basename(prompt_file)))