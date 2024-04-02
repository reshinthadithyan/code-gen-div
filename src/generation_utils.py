import json
from transformers import AutoModel
from numpy.linalg import norm
import torch
from tqdm import tqdm
import os
import datasets

# from src.metrics import get_centroid, sum_squared_errors, average_pairwise_distance, silhouette_coefficient

EMBD_PATH = "embd_path"

def load_json_path(path):
    with open(path, "r") as f:
        return json.load(f)


def get_human_eval_ds():
    return datasets.load_dataset("openai_humaneval",split="test")

class Embd:
    def __init__(self, generation_dict_path:str,embd_file_name:str,device:str="cuda",remove_prompt:bool=True):
        self.embd_file_path = os.path.join(EMBD_PATH,embd_file_name)
        self.generation_dict = load_json_path(generation_dict_path)
        self.num_solution = len(self.generation_dict)
        self.num_samples_per_solution = len(self.generation_dict[0])
        self.embd_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-code",trust_remote_code=True).to("cuda")
        self.vector_list = {}
        self.cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
        self.human_eval_benchmark = get_human_eval_ds()
        self.device = device
        self.remove_prompt : bool = remove_prompt


    def remove_duplicates(self,generation_list):
        """ dirty way of removing exact duplicates from a list of strings """
        return list(set(generation_list))
         
    def remove_duplicates_loop(self,generation_dict):
        for i in range(len(generation_dict)):
            generation_dict[i] = self.remove_duplicates(generation_dict[i])
            if self.remove_prompt:
                for j in range(len(generation_dict[i])):                    
                    generation_dict[i][j]=generation_dict[i][j].split('"""')[-1]

        self.generation_dict = generation_dict
        return generation_dict

    def embd_all_solutions(self):
        if os.path.exists(self.embd_file_path):
            print("Loading from existing embd file")
            self.vector_list = torch.load(self.embd_file_path)
            return self.vector_list
        else:
        
            for i in tqdm(range(self.num_solution),leave=False):
                soln_list = []
                for j in tqdm(range(len(self.generation_dict[i])),leave=False):
                    sample = self.generation_dict[i][j]
                    embd_sample = self.embd_model.encode([sample])
                    embd_sample_torch = torch.from_numpy(embd_sample)
                    soln_list.append(embd_sample_torch)
                self.vector_list[i] = torch.stack(soln_list).squeeze(1)
            torch.save(self.vector_list,self.embd_file_path)
            return self.vector_list


    
    
    


if __name__ == "__main__":
    dummy_generation_path = "/weka/ckpts/stablecode_modelablations/deepseek-ai/deepseek-coder-6.7b-base/{py}_gens_temp_0.2.json"
    embd_class = Embd(dummy_generation_path,                   
                      embd_file_name="ds_6.7b_temp=0.2_embd.pt"
    )
    embd_class.remove_duplicates_loop(embd_class.generation_dict)
    embd_vector = embd_class.embd_all_solutions()
