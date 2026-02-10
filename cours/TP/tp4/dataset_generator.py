import asyncio
import random
import sys
from dataset_utils import load_csv, save_prompt_responses, load_prompt_responses
from promptotron import Promptotron


class DatasetGenerator :
    def __init__(self, api_key: str, base_url: str, dino_dataset_path: str) :
        self.client = Promptotron(api_key, base_url)
        self.dino_dataset = load_csv(dino_dataset_path)


    def generate_dino_description(self, shuffle = False, max = sys.maxsize, start_index = 0) :
        length = min(len(self.dino_dataset), max + start_index)

        low_temp_dataset = []
        high_temp_dataset = []

        batch_size = 50

        if shuffle :
            data = self.dino_dataset.copy()
            random.shuffle(data)
        else :
            data = self.dino_dataset

        try:
            for i in range(start_index, length, batch_size):
                print(f"Batch {i} → {min(i+batch_size, length)}")

                batch_results = asyncio.run(
                    self.generate_dinos_batch(data, i, batch_size)
                )

                for low_res, high_res in batch_results:
                    low_temp_dataset.append(low_res)
                    high_temp_dataset.append(high_res)
        finally:
            return {
                "low_temp" : low_temp_dataset,
                "high_temp" : high_temp_dataset
            }
        
    async def generate_one_dino(self, dino, i):
        print(f'{dino["common_name"]}')

        input = (
            f'create a description of the following dinosaurs species : '
            f'{dino["common_name"]}, give the size, the period range, '
            f'the fossiles location, a physical decription, '
            f'the fossiles records, a diet description '
            f'and a description of the environnement. Dont use markdown'
        )

        return await self.client.double_temperature_prompt_async(input, i)
    
    async def generate_dinos_batch(self, data, start_index, batch_size=5):
        tasks = []

        for i in range(start_index, min(start_index + batch_size, len(data))):
            tasks.append(self.generate_one_dino(data[i], i))

        return await asyncio.gather(*tasks)


    def generate_and_save_dino_description(self, low_temp_path: str, hight_temp_path: str, shuffle = False, max = sys.maxsize, start_index = 0) :
        dataset = self.generate_dino_description(shuffle, max, start_index)
        save_prompt_responses(dataset["low_temp"], low_temp_path)
        save_prompt_responses(dataset["high_temp"], hight_temp_path)

            
    def generate_and_append_dino_description(self, low_temp_path: str, hight_temp_path: str, shuffle = False, max = sys.maxsize, start_index = 0) :
        dataset = self.generate_dino_description(shuffle, max, start_index)
        low_dataset = load_prompt_responses(low_temp_path)
        high_dataset = load_prompt_responses(hight_temp_path)

        low_dataset = list({item.id: item for item in (low_dataset + dataset["low_temp"])}.values())
        high_dataset = list({item.id: item for item in (high_dataset + dataset["high_temp"])}.values())

        save_prompt_responses(low_dataset, low_temp_path)
        save_prompt_responses(high_dataset, hight_temp_path)


    def generate_all_dinos_description(self, low_temp_path: str, hight_temp_path: str) :
        current_index = len(load_prompt_responses(low_temp_path))
        length = len(self.dino_dataset)
        while current_index < length :
            print(f'current index = {current_index}')
            self.generate_and_append_dino_description(low_temp_path, hight_temp_path, start_index=current_index)
            current_index = len(load_prompt_responses(low_temp_path))

        print("All dataset created")

