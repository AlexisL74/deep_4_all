import os
import random
from dotenv import load_dotenv
from dataset_generator import DatasetGenerator
from dataset_utils import load_csv, save_prompt_responses, load_prompt_responses



if __name__ == "__main__":
    low = load_prompt_responses("data/low_test_final.json")
    high = load_prompt_responses("data/high_test_final.json")

    print(low[0])

    save_prompt_responses(low, "data/low_temperature_dataset.json")
    save_prompt_responses(high, "data/high_temperature_dataset.json")