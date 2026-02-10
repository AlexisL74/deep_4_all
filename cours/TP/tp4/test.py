import os
from dotenv import load_dotenv
from dataset_utils import load_csv, save_prompt_responses, load_prompt_responses
from Promptotron import Promptotron



if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("API_URL")
    
    proptotron = Promptotron(api_key, base_url)
    response = proptotron.low_temperature_prompt("salut ma couille")
    save_prompt_responses([response], "./")