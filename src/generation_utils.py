import json


def load_json_path(path):
    with open(path, "r") as f:
        return json.load(f)



if __name__ == "__main__":
    dummy_generation_path = "/weka/home-reshinth/work/benchmarks/bigcode-evaluation-harness/generations_humaneval.json"
    dummy_generation = load_json_path(dummy_generation_path)
    print(dummy_generation[0])