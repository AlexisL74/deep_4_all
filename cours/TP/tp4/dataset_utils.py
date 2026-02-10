import csv


def load_csv(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)
    
__all__  = [ load_csv ]