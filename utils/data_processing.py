from typing import List
import re
import os
import pandas as pd

def extract_questions(text):
    """
    Extracts questions from a given text using a regular expression.

    Parameters:
        text (str): The input text to extract questions from.

    Returns:
        list: A list of questions extracted from the text.
    """
    pattern = r'(?:Question:\s*)?(.*?)(?:\n|$)'  # Match questions, optionally starting with "Question:"
    questions = re.findall(pattern, text)
    questions = [q.split('Question: ')[-1] for q in questions]  # Remove leading "Question: " if present
    return questions

def process_questions(df):
    """
    Processes the dataframe by extracting, cleaning, and exploding the questions.

    Parameters:
        df (pd.DataFrame): The dataframe containing the 'generated_questions' column.

    Returns:
        pd.DataFrame: A dataframe with processed questions.
    """
    # Extract questions from generated questions
    df['questions'] = df.generated_questions.apply(extract_questions)
    
    # Clean up questions by removing empty ones
    df['questions'] = df.questions.apply(lambda x: [y for y in x if len(y) > 0])

    # Add a column for the number of questions
    df['num_questions'] = df.questions.apply(lambda x: len(x))

    # Drop the original 'generated_questions' column
    df.drop('generated_questions', axis=1, inplace=True)

    # Explode the 'questions' list so that each question gets its own row
    df_out = df.explode('questions')

    # Add an image_idx column and filter out invalid types
    df_out['image_idx'] = df_out.index
    df_out = df_out[df_out.questions.apply(lambda x: isinstance(x, str))]

    # Add question indices based on image_idx
    df_out['question_idx'] = df_out.groupby('image_idx').cumcount().values

    # Reset index
    df_out.reset_index(inplace=True, drop=True)
    
    return df_out

def extract_questions(text):
    """
    Extracts questions from a given text using a regular expression.
    """
    pattern = r'(?:Question:\s*)?(.*?)(?:\n|$)'
    questions = re.findall(pattern, text)
    questions = [q.split('Question: ')[-1] for q in questions]
    questions = [re.sub(r'^\d+\.\s', '', q) for q in questions]
    return questions

def save_df(input_list: List[List[str]], 
            column_list: List[str],
            file_name: str, 
            output_path: str,
            process_data: bool = False):
    """
    Saves the dataframe to a CSV file. Optionally processes the data before saving.
    """
    inputs = {col: values for col, values in zip(column_list, input_list)}
    df = pd.DataFrame(inputs)
    if process_data:
        df = process_questions(df)
    df.to_csv(os.path.join(output_path, file_name), index=False)


class QuestionParser:
    """
    A class used to parse and process the difficult questions from a dataset.

    Attributes:
        difficulty_threshold (int): The threshold to determine if a question is difficult.
        df_path (str): The path to the dataframe CSV file.
        df (pd.DataFrame): The dataframe loaded from the CSV file.

    Methods:
        build_difficult_df(): Builds a dataframe containing a list of only the difficult questions for each image.
    """
    def __init__(self, difficulty_threshold: int, df_path: str):
        self.difficulty_threshold = difficulty_threshold
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        # Preserve original row order
        self.df["_row_ordinal"] = range(len(self.df))

    def build_difficult_df(self):
        # Drop duplicates but keep image_idx
        df_out = (
            self.df.drop(columns=['vlm_answers', 'questions', 'judgments', '_row_ordinal'], errors='ignore')
            .drop_duplicates('image_idx')
            .reset_index(drop=True)
        )
        # Count judgments < 2
        grouped = (
            self.df.groupby(['image_idx', 'questions'])['judgments']
            .apply(lambda x: (x < 2).sum())
            .reset_index(name='count_judgments_smaller_than_2')
        )
        grouped['is_difficult'] = grouped['count_judgments_smaller_than_2'] >= self.difficulty_threshold
        # Build a dictionary to mark which questions are difficult
        difficulty_lookup = {
            (row.image_idx, row.questions): row.is_difficult for _, row in grouped.iterrows()
        }
        # Gather difficult questions in original order
        from collections import defaultdict
        image_to_questions = defaultdict(list)
        for _, row in self.df.sort_values('_row_ordinal').iterrows():
            if difficulty_lookup.get((row.image_idx, row.questions), False):
                if row.questions not in image_to_questions[row.image_idx]:
                    image_to_questions[row.image_idx].append(row.questions)
        # Create final DataFrame
        difficult_df = pd.DataFrame([
            {'image_idx': idx, 'questions': qlist}
            for idx, qlist in image_to_questions.items()
        ])
        # Merge with original columns
        df_out = pd.merge(df_out, difficult_df, on='image_idx', how='left')
        return df_out