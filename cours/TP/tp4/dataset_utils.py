import csv
from dataclasses import asdict
import json
<<<<<<< HEAD
from Promptotron import PrompteResponse
from data_class.DASResult import DASResult
=======
from promptotron import PrompteResponse

>>>>>>> 36c990fc10da8ad3c84f24cefcecd64145102417

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

def save_das_results(responses: list, path: str) -> None:
    payload = [asdict(r) for r in responses]

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

def das_results_to_sharegpt(results: list) -> list[dict]:
    sharegpt = []
    for r in results:
        sharegpt.append({
            "conversations": [
                {"from": "human", "value": r.prompt},
                {"from": "gpt", "value": r.teacher_response},
            ]
        })
    return sharegpt

def save_sharegpt(results: list, path: str) -> None:
    sharegpt = das_results_to_sharegpt(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sharegpt, f, ensure_ascii=False, indent=2)
    print(f"[ShareGPT] {len(sharegpt)} conversations sauvegardées dans '{path}'")
    
__all__ = [load_csv, save_prompt_responses, load_prompt_responses, save_das_results, das_results_to_sharegpt, save_sharegpt]