import os
import pandas as pd
import json
import numpy as np
import argparse
import time
import logging
from tqdm.contrib import tzip
from tqdm import tqdm
from ast import literal_eval
import google.generativeai as genai
from google.generativeai import GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
A script for evaluation of dense captions, based on proposition decomposition.
The procedure is as follows:
    1. Generate propositions for the ground truth descriptions, by flagging the `generate_gt_propositions` argument as True.
       - Use the 'with_gemini_captions' flag to determine whether to evaluate on DOCCI or on synthetic captions based on your own df_path. Otherwise, will default to DOCCI.
    2. Evaluate the generated propositions based on the ground truth captions and propositions.
       - Use the 'gt_propositions_path' argument to provide the path to the ground truth propositions generated in the previous step.
       - The 'df_path' argument should point to the CSV file with the following columns: [generated_description, original_description]
"""

def configure_genai():
    """Configure the GenAI environment and API key."""
    GEMINI_MODEL_CARD = 'gemini-1.5-flash-002'
    genai.configure(api_key=os.environ['API_KEY'])
    return GEMINI_MODEL_CARD

def create_output_json():
    """Create the initial output JSON structure."""
    return {
        "mean_entailed_propositions_recall": 0,
        "mean_entailed_propositions_precision": 0,
        "mean_contradicted_propositions_recall": 0,
        "mean_contradicted_propositions_precision": 0,
        "mean_neutral_propositions_recall": 0
    }

def create_system_descriptions():
    """Create system descriptions for propositions and judgment."""
    SYSTEM_DESC = """Decompose the given image caption into clear and simple propositions, ensuring they are interpretable out of context. 

Follow these guidelines:
1. Split compound sentences into simple sentences.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the propositions by:
    - Adding necessary modifiers to nouns or entire sentences to clarify context.
    - Replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entity they refer to, ensuring no references rely on prior information.
4. If there are any contradicting propositions, present both as separate propositions.
5. Present the results in JSON format with the following structure:
    - `"propositions"`: an array of objects, each containing:
      - `"id"`: a unique number for each proposition, starting from 1.
      - `"proposition"`: the decomposed, decontextualized proposition as a string.
Only use the json format, without trailing \n, ```, or the word JSON etc. Make sure you use signle " quotes and not double "" in the json output when representing strings.
It is extremely important to have the right JSOM format, otherwise the evaluation will fail, as is shown in the example below.

Example:

Input:
The image shows a concrete sidewalk. A diagonally oriented rectangular section of the sidewalk is textured with 17 parallel lines. To the right of the textured section, a red stripe runs parallel to the edge of the picture. The words "FIRE LANE" are inscribed within the stripe, with the top of the letters oriented toward the top right of the image. To the left of the textured section, the word "ROW" is written in orange. Above the "ROW", an orange line, parallel to the top edge of the textured section, bisects a small orange circle.

Output:
{
  "propositions": [
    { "id": 1, "proposition": "There is a sidewalk in the image." },
    { "id": 2, "proposition": "The sidewalk is made of concrete." },
    { "id": 3, "proposition": "There is a rectangular section of the sidewalk." },
    { "id": 4, "proposition": "The rectangular section is oriented diagonally." },
    { "id": 5, "proposition": "The rectangular section is textured with lines." },
    { "id": 6, "proposition": "There are 17 parallel lines on the rectangular section." },
    { "id": 7, "proposition": "There is a red stripe in the image." },
    { "id": 8, "proposition": "The red stripe is to the right of the textured section." },
    { "id": 9, "proposition": "The red stripe runs parallel to the edge of the picture." },
    { "id": 10, "proposition": "The words 'FIRE LANE' are written." },
    { "id": 11, "proposition": "The words 'FIRE LANE' are inscribed within the stripe." },
    { "id": 12, "proposition": "The top of the letters of the words 'FIRE LANE' is oriented toward the top right of the image." },
    { "id": 13, "proposition": "There is text to the left of the textured section." },
    { "id": 14, "proposition": "The word 'ROW' is written." },
    { "id": 15, "proposition": "The word 'ROW' is written in orange." },
    { "id": 16, "proposition": "There is an orange line above the word 'ROW'." },
    { "id": 17, "proposition": "The orange line is parallel to the top edge of the textured section." },
    { "id": 18, "proposition": "There is a small orange circle in the image." },
    { "id": 19, "proposition": "The orange line bisects the small orange circle." }
  ]
}"""

    SYSTEM_JUDGE = """You are given a ground truth image description and a list of propositions. Your task is to analyze each proposition and check whether it is entailed by the ground truth image description. A proposition is considered entailed if all the information in it can be inferred from the ground truth description. If the proposition introduces new information that is not present in the ground truth or contradicts it, it is not entailed.

Additional Criteria:
1. If the proposition contains neutral information (subjective information, such as describing the environment as "lively" or "pleasant"), it should not be counted as either entailed or contradicted. Instead, it should be judged as "Neutral."
2. If the proposition introduces additional visual information that is not in the ground truth, it should be judged as "Contradicted."

For each proposition, respond with the proposition number and its corresponding judgment as either "Entailed," "Contradicted," or "Neutral." Your output should maintain the same number of propositions as the input list.

Provide the results in the following JSON format:
- `"propositions"`: an array of objects, each containing:
  - `"id"`: the number of the proposition.
  - `"judgment"`: the result for each proposition as "Entailed," "Contradicted," or "Neutral."
- `"summary"`: an object containing:
  - `"contradicting_count"`: the number of contradicting propositions.
  - `"entailed_count"`: the number of entailed propositions.
  - `"neutral_count"`: the number of neutral propositions.

If the list of propositions is empty, provide an empty json as the output, with the same format.

Example output:

{
  "propositions": [
    { "id": 1, "judgment": "Entailed" },
    { "id": 2, "judgment": "Contradicted" },
    { "id": 3, "judgment": "Neutral" },
    { "id": 4, "judgment": "Entailed" }
  ],
  "summary": {
    "contradicting_count": 1,
    "entailed_count": 2,
    "neutral_count": 1
  }
}"""
    return SYSTEM_DESC, SYSTEM_JUDGE

def create_response_schemas():
    """Create response schemas for propositions and judgment."""
    RESPONSE_SCHEMA_JUDGE = {
        "type": "OBJECT",
        "properties": {
            "propositions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER", "nullable": False},
                        "judgment": {"type": "STRING", "enum": ["Entailed", "Contradicted", "Neutral"], "nullable": False},
                    },
                    "required": ["id", "judgment"],
                },
            },
            "summary": {
                "type": "OBJECT",
                "properties": {
                    "contradicting_count": {"type": "INTEGER", "nullable": False},
                    "entailed_count": {"type": "INTEGER", "nullable": False},
                    "neutral_count": {"type": "INTEGER", "nullable": False},
                },
                "required": ["contradicting_count", "entailed_count", "neutral_count"],
            },
        },
        "required": ["propositions", "summary"],
    }

    RESPONSE_SCHEMA_DESC = {
        "type": "OBJECT",
        "properties": {
            "propositions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER", "nullable": False},
                        "proposition": {"type": "STRING", "nullable": False},
                    },
                    "required": ["id", "proposition"],
                },
            },
        },
        "required": ["propositions"],
    }
    return RESPONSE_SCHEMA_JUDGE, RESPONSE_SCHEMA_DESC

GEMINI_MODEL_CARD = configure_genai()
OUTPUT_JSON = create_output_json()
SYSTEM_DESC, SYSTEM_JUDGE = create_system_descriptions()
RESPONSE_SCHEMA_JUDGE, RESPONSE_SCHEMA_DESC = create_response_schemas()

def check_csv_and_count_rows(csv_path):
    """
    Checks if the CSV exists, which means that generation should continue from last row,
    and counts its rows, creating the directory if necessary.
    """
    directory = os.path.dirname(csv_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory created: {directory}")
        return None
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        num_rows = len(df)
        return (
            num_rows,
            df.entailed_propositions.tolist(),
            df.contradicted_propositions.tolist(),
            df.neutral_propositions.tolist(),
            df.propositions.apply(literal_eval).tolist(),
            df.total_propositions.tolist(),
        )
    else:
        logger.info(f"CSV file does not exist: {csv_path}, starting to evaluate from scratch")
        return 0, [], [], [], [], []

class Postprocess:
    """Class for postprocessing the dataframe to extract propositions and judgments and save as CSV."""
    def __init__(self, output_path):
        self.output_path = f"{os.path.splitext(output_path)[0]}_parsed.csv"
    
    def postprocess(self, df):
        """Postprocess the dataframe to extract propositions and judgments and save as CSV."""
        propositions = df.propositions.apply(lambda x: x['propositions']).tolist()

        ids, propositions_list, judgment_list = [], [], []

        for i in range(len(propositions)):
            for prop in propositions[i]:
                ids.append(prop['id'])
                propositions_list.append(prop['proposition'])
                judgment_list.append(prop['judgment'])

        df_out = pd.DataFrame({
            'id': ids,
            'proposition': propositions_list,
            'judgment': judgment_list,
        })
        df_out.to_csv(self.output_path, index=False)

class GeminiModel:
    """Class for interacting with the Gemini model."""
    def __init__(self, system, response_schema=RESPONSE_SCHEMA_DESC):
        self.generation_config = GenerationConfig(
            top_k=1, 
            top_p=0,
            response_mime_type="application/json",
            response_schema=response_schema
        )
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_CARD, 
            system_instruction=[system], 
            generation_config=self.generation_config
        )

    def _create_propositions_prompt(self, description):
        """Creates a prompt for the Gemini model based on the input description."""
        return f"Input:\n{description}\n\nOutput:" 

    def _create_judgment_prompt(self, description, propositions):
        """Creates a prompt for the Gemini model based on the input description and propositions."""
        return f"Ground truth image description:\n{description}\nPropositions:\n{propositions}"

    def _generate_answer(self, prompt):
        """Generates a response using the Gemini model based on a prompt."""
        chat = self.model.start_chat()
        return chat.send_message(
            content={"role": "user", "parts": prompt},
            generation_config=self.generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
            }
        ).text.strip()
    
    def get_response(self, prompt):
        """Attempts to generate a response, with retry logic for failures in case of too many requests."""
        try:
            return self._generate_answer(prompt)
        except Exception as e:
            logger.error(e)
            time.sleep(120)
            try:
                return self._generate_answer(prompt)
            except Exception:
                return ""

class GetMetrics:
    """Class for calculating and printing metrics for the propositions."""
    def __init__(self, args):
        self.df_p = pd.read_csv(args.df_path)
        self.df_p['propositions'] = self.df_p.gt_propositions.apply(lambda x: literal_eval(x)['propositions'])
        self.output_json = OUTPUT_JSON
    
    def print_ratio(self, df):
        """Prints the recall, precision, and average metrics for the propositions."""
        self.output_json['mean_entailed_propositions_recall'] = (df.entailed_propositions / self.df_p.iloc[:len(df)].total_propositions).mean()
        self.output_json['mean_entailed_propositions_precision'] = (df.entailed_propositions / df.total_propositions).mean()
        self.output_json['mean_contradicted_propositions_recall'] = (df.contradicted_propositions / self.df_p.iloc[:len(df)].total_propositions).mean()
        self.output_json['mean_contradicted_propositions_precision'] = (df.contradicted_propositions / df.total_propositions).mean()
        self.output_json['mean_neutral_propositions_recall'] = (df.neutral_propositions / self.df_p.iloc[:len(df)].total_propositions).mean()

        logger.info(f"Iteration {len(df)}:\n")
        logger.info(self.output_json)

def generate_gt_propositions(args):
    """Generates propositions for the ground truth descriptions."""
    df = pd.read_csv(args.df_path)
    gt_propositions_path = os.path.join(args.output_dir, 'gt_propositions.csv')
    if gt_propositions_path is not None and os.path.exists(gt_propositions_path):
        logger.info("Ground truth propositions already exist, adding to current df")
        df_gt = pd.read_csv(gt_propositions_path)
        df['gt_propositions'] = df_gt.gt_propositions.tolist()[:len(df)]
        df.to_csv(args.df_path, index=False)
    else:
        descriptions = df.original_description.tolist()
        gemini = GeminiModel(system=SYSTEM_DESC)

        propositions = []
        total_propositions = []

        for description in tqdm(descriptions, desc="Generating propositions for the ground truth descriptions"):
            prompt = f"Input:\n{description}\n\nOutput:"
            generated_propositions = gemini.get_response(prompt)
            generated_propositions = json.loads(generated_propositions)

            propositions.append(generated_propositions)
            total_propositions.append(len(generated_propositions['propositions']))

            if len(propositions) % args.log_every == 0:
                df_out = pd.DataFrame({"gt_propositions" : propositions, "total_propositions" : total_propositions})
                df_out.to_csv(gt_propositions_path, index=False)

        df_out = pd.DataFrame({"gt_propositions" : propositions, "total_propositions" : total_propositions})
        df_out.to_csv(gt_propositions_path, index=False)   
        df['gt_propositions'] = propositions
        df['total_propositions'] = total_propositions
        df.to_csv(args.df_path, index=False)

def eval_prop(args):
    """Evaluates the propositions based on input arguments and processes the results."""
    os.makedirs(args.output_dir, exist_ok=True)
    generate_gt_propositions(args)

    output_path = os.path.join(args.output_dir, 'generated_descriptions_propositions.csv')
    metrics_output_path = os.path.join(args.output_dir, 'propositions_metrics.json')
    postprocess = Postprocess(output_path)
    df = pd.read_csv(args.df_path)

    metric = GetMetrics(args)
    descriptions, ground_truths = df.generated_description.tolist(), df.original_description.tolist()

    gemini = GeminiModel(system=SYSTEM_DESC, response_schema=RESPONSE_SCHEMA_DESC)
    gemini_judge = GeminiModel(system=SYSTEM_JUDGE, response_schema=RESPONSE_SCHEMA_JUDGE)

    cur_lists = check_csv_and_count_rows(output_path)
    i, entailed_propositions, contradicted_propositions, neutral_propositions, propositions, total_propositions = cur_lists

    for g, d in tzip(ground_truths[i:], descriptions[i:], desc="Evaluating propositions"):
        prompt = f"Input:\n{d}\n\nOutput:"
        generated_props = gemini.get_response(prompt)

        prompt = f"Ground truth image description:\n{g}\nPropositions:\n{generated_props}"
        judgment = gemini_judge.get_response(prompt)
        try:
            judgment = json.loads(judgment)
            entailed_prop = judgment['summary']['entailed_count']
            neutral_prop = judgment['summary']['neutral_count']
            contradicted_prop = judgment['summary']['contradicting_count']
            total_prop = entailed_prop + neutral_prop + contradicted_prop
        except Exception:
            logger.error("Error loading judgment JSON")
            logger.error(judgment)
            entailed_prop = neutral_prop = contradicted_prop  = total_prop = -1
        try:
            generated_props = json.loads(generated_props)
        except:
            logger.error("Error in loading generated propositions JSON")
            logger.error(generated_props)
        for gen_prop, judg_prop in zip(generated_props['propositions'], judgment['propositions']):
            gen_prop['judgment'] = judg_prop['judgment']

        entailed_propositions.append(entailed_prop)
        neutral_propositions.append(neutral_prop)
        contradicted_propositions.append(contradicted_prop)
        propositions.append(generated_props)
        total_propositions.append(total_prop)
        i += 1
        if i % args.log_every == 0:
            df_out = save_progress(i, descriptions, ground_truths, entailed_propositions, contradicted_propositions, neutral_propositions, propositions, total_propositions, output_path)
            metric.print_ratio(df_out)
            postprocess.postprocess(df_out)

    df_out = save_progress(i, descriptions, ground_truths, entailed_propositions, contradicted_propositions, neutral_propositions, propositions, total_propositions, output_path)
    metric.print_ratio(df_out)
    postprocess.postprocess(df_out)
    with open(metrics_output_path, 'w') as fp:
        json.dump(metric.output_json, fp)

def save_progress(i, descriptions, ground_truths, entailed_propositions, contradicted_propositions, neutral_propositions, propositions, total_propositions, output_path):
    """Saves the progress of the evaluation to a CSV file."""
    df_out = pd.DataFrame({
        'generated_description': descriptions[:i],
        'original_description': ground_truths[:i],
        'entailed_propositions': entailed_propositions,
        'contradicted_propositions': contradicted_propositions,
        'neutral_propositions': neutral_propositions,
        'propositions': propositions,
        'total_propositions': total_propositions
    })
    df_out.to_csv(output_path, index=False)
    return df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, required=True,
                        help="Path to the CSV file with the following columns: [generated_description, original_description]")
    parser.add_argument('--output_dir', default='./outputs/eval_results/', type=str,
                        help="Path to save the evaluation results")
    parser.add_argument('--log_every', default=100, type=int,
                        help="Write to CSV every n iterations")
    args = parser.parse_args()

    logger.info("\n\nRunning inference for: %s", args)

    eval_prop(args)