import csv
from dataclasses import asdict
import json
from promptotron import PrompteResponse


def load_csv(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)
    
def save_prompt_responses(responses: list[PrompteResponse], path: str) :
    payload = [asdict(r) for r in responses]

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2
        )

def load_prompt_responses(path: str) :
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return [PrompteResponse(**item) for item in data]
    except:
        return []
    
__all__  = [ load_csv, save_prompt_responses, load_prompt_responses ]