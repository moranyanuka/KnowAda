# Bridging the Visual Gap: Fine Tuning Multimodal Models With Knowledge Adapted Captions

**NAACL 2025** 

<a href="https://scholar.google.com/citations?user=ZsXf6OMAAAAJ&hl=en">Moran Yanuka</a>,
<a href="https://assafbk.github.io/website/">Assaf Ben-Kish</a>,
<a href="https://yonatanbitton.github.io/">Yonatan Bitton</a>,
<a href="https://sites.google.com/site/idanszpektor">Idan Szpektor</a>,
<a href="https://www.giryes.sites.tau.ac.il/">Raja Giryes</a>


<a href="https://arxiv.org/abs/2411.09018"><img src="https://img.shields.io/badge/arXiv-2411.09018-b31b1b.svg"></a>

## Setup

### Clone Project
```Shell
git clone https://github.com/moranyanuka/knowada.git
cd knowada
```

### Create the Environment
To set up our environment, please run:
```Shell
conda env create -f environment.yml
conda activate knowada
```

Add your Gemini API key:
```Shell
export API_KEY='<your-key>'
```

## DNLI Dense Caption Evaluation

### Running the *DNLI* evaluation of dense captions for your VLM

First, create a CSV file with the following columns:

- `original_description`: Contains the ground-truth image description from the evaluation dataset
- `generated_description`: Contains the generated description of the VLM to evaluate

See an example of such a file [here](https://github.com/moranyanuka/KnowAda/blob/main/examples/model_generation_sample.csv).

Then, run the following command:
```Shell
python eval/generate_propositions.py \
       --df_path <path-to-model-generation> \
       --output_dir <path-to-evaluation-output>
```

The script will write the propositions of the ground truth descriptions, the propositions of the generated descriptions, and the final metrics to `output_dir`.

## KnowAda

To rewrite the DOCCI captions according to the knowledge gaps of PaliGemma, run the following script:
```Shell
python run.py \
       --generate_questions True \
       --generate_answers True \
       --generate_judgments True \
       --generate_rewritten_descriptions True \
       --output_folder <path-to-output-directory>
```

This will generate the following files:

- `questions.csv`: Contains the generated questions based on the image descriptions
- `answers.csv`: The VLM's sampled answers to each question
- `judgments.csv`: The judgments determining whether a given answer is correct on a scale of 1-3 (1 is completely incorrect, 3 is completely correct)
- `difficult_questions_list.csv`: Contains for each descriptions, all the questions that are considered unknown for a given threshold
- `rewritten_captions.csv`: The final rewritten captions based on the unknown questions

You can adjust some of the parameters in each stage of the pipeline using the [config files](https://github.com/moranyanuka/KnowAda/tree/main/configs) (e.g., the train/test split, the difficulty_threshold for determining if a question is unknown, etc.)