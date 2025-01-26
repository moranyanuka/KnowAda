import logging

from utils.utils import (parse_args,
                         load_config)

from tasks.question_generation import generate_questions
from tasks.vlm_answer_generation import generate_answers
from tasks.question_judgment_generation import judge_questions
from tasks.caption_rewriting_generation import rewrite_captions


# Suppress logging warnings
import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    if args.generate_questions:
        config = load_config('configs/question_generation_config.json')
        logging.info(f"generating questions for {config['dataset']}\n")
        generate_questions(config, output_path=args.output_folder)
        logging.info(f"Done generating questions\n")

    if args.generate_answers:
        config = load_config('configs/vlm_answer_generation_config.json')
        logging.info(f"Sampling VLM answers for {config['dataset']}\n")
        generate_answers(config, output_path=args.output_folder)
        logging.info(f"Done generating answers\n")
    
    if args.generate_judgments:
        config = load_config('configs/answer_judgment_config.json')
        logging.info(f"judging answers to the questions for {config['dataset']}\n")
        judge_questions(config, output_path=args.output_folder)
        logging.info(f"Done judging answers\n")
    
    if args.generate_rewritten_descriptions:
        config = load_config('configs/caption_rewrite_config.json')
        logging.info(f"rewriting descriptions for {config['dataset']}\n")
        rewrite_captions(config, output_path=args.output_folder)
        logging.info(f"Done rewriting captions\n")
