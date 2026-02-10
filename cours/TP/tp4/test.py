import os
import random
from dotenv import load_dotenv
from dataset_generator import DatasetGenerator
from dataset_utils import load_csv, save_prompt_responses, load_prompt_responses



if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_URL")

    dataset_path = "data/dino_dataset.csv"

    generator = DatasetGenerator(api_key, base_url, dataset_path)
    generator.generate_all_dinos_description("data/low_test_final", "data/high_test_final")