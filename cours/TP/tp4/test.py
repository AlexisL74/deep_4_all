import os
import random
from dotenv import load_dotenv
from dataset_utils import load_csv, save_prompt_responses, load_prompt_responses
from Promptotron import Promptotron



if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_URL")

    dino_dataset = load_csv("data/dino_dataset.csv")

    low_temp_responses = []
    high_temp_responses = []

    proptotron = Promptotron(api_key, base_url)

    random.shuffle(dino_dataset)

    length = len(dino_dataset)
    max = min(len(dino_dataset), 10)
    for i in range(0, max) :
        dino = dino_dataset[i]
        print(f'{i+1}/{max} : {dino["common_name"]}')

        input = f'describe the following dinosaures species : {dino["common_name"]}'

        low_res = proptotron.low_temperature_prompt(input)
        high_res = proptotron.high_temperature_prompt(input)
        low_temp_responses.append(low_res)
        high_temp_responses.append(high_res)

    save_prompt_responses(low_temp_responses, "data/low_temperature_responses.json")
    save_prompt_responses(high_temp_responses, "data/high_temperature_responses.json")