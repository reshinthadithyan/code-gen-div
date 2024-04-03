import gradio as gr
import numpy as np
import torch
import datasets
import json


def read_stats_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)



if __name__ == "__main__":
    humaneval_ds = datasets.load_dataset("openai_humaneval")
    stats_json = read_stats_json("stats.json")

    




