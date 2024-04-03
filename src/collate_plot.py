import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
from plot import load_torch_tensor, create_labels, process_and_plot_pca
from metrics import get_centroid, sum_squared_errors, average_pairwise_distance, silhouette_coefficient



def split_different_paths(embd_path):

    embd_path_list =[ os.path.join(embd_path,i) for i in  os.listdir(embd_path) ]
    return embd_path_list

    


def write_stats_to_file(stats_dict_map,json_path):
    json_path = os.path.join(os.getcwd(), json_path)
    with open(json_path, "w") as f:
        json.dump(stats_dict_map, f)

def get_stats(embd_dict,prob_index):
    embd_tensor = embd_dict[prob_index]
    centroids = get_centroid(embd_tensor)
    sse = sum_squared_errors(embd_tensor, centroids)
    apd = average_pairwise_distance(embd_tensor)
    sc = silhouette_coefficient(embd_tensor, centroids, torch.zeros(embd_tensor.shape[0]))
    return centroids, sse, apd, sc




def main(args):

    embd_paths = split_different_paths(args.embd_path)

    embd_dict_map = {}
    stats_dict_map = {}
    for embd_path in tqdm(embd_paths, desc="Processing Embedding Files",leave=False):
        embd_dict_map[embd_path] = load_torch_tensor(embd_path)
        stats_dict = {
                # "centroids": [],
                "sse": [],
                "apd": [],
                "sc": []
            }
        for prob_index in tqdm(embd_dict_map[embd_path], desc="Processing Problems",leave=False ):
            centroids, sse, apd, sc = get_stats(embd_dict_map[embd_path], prob_index)
            # stats_dict["centroids"].append(centroids)
            stats_dict["sse"].append(sse.item())
            stats_dict["apd"].append(apd.item())
            stats_dict["sc"].append(sc)
        #Average sse,apd
        print(embd_path.split("/")[-1].replace(".pt", "") + " Stats")
        print(f"Average sse: {np.mean(stats_dict['sse'])}")
        print(f"Average apd: {np.mean(stats_dict['apd'])}")
        print("####")
        stats_dict_map[embd_path] = stats_dict

    write_stats_to_file(stats_dict_map, "collated_stats/stats_stripped.json")
    # Average all stats per embd
    avg_stats_dict = {}
    for embd_path in stats_dict_map:
        avg_stats_dict[embd_path] = {
            # "centroids": torch.stack(stats_dict_map[embd_path]["centroids"]).mean(dim=0),
            "sse": np.mean(stats_dict_map[embd_path]["sse"]),
            "apd": np.mean(stats_dict_map[embd_path]["apd"]),
            "sc": np.mean(stats_dict_map[embd_path]["sc"])
        }
    
    def plot_ind_prob(embd_dict_map,prob_index):
        x = []
        labels = []
        for embd_path in embd_dict_map:
            embd_tensor = embd_dict_map[embd_path][prob_index]
            label = [embd_path.split("/")[-1].replace(".pt","") for i in range(embd_tensor.shape[0])]
            labels.extend(label)
            x.append(embd_tensor)
        
        x = torch.cat(x).numpy()
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(x)
        reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
        #Add labels to the reduced_df
        reduced_df['labels'] = labels
        # Plotting
        plt.figure(figsize=(10, 10))
        sns.color_palette("flare")
        sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue=labels)
        #Add title
        plt.title(f'PCA projection of problem {prob_index} across different model gens')
        plot_save_path = f"plots/probwise/problem_{prob_index}_pca_plot.png"
        plt.savefig(plot_save_path)
        plt.close()
        

    for i in tqdm(range(163)):
        plot_ind_prob(embd_dict_map, i)


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--embd_paths", default="embd_path/ds_33b_inst_temp=0.2_embd.pt,embd_path/ds_1.3B_temp=0.2_embd.pt,embd_path/mistral_7B_inst_v2_temp=0.2_embd.pt,embd_path/sc_3b_temp=0.2_embd.pt,embd_path/sc_inst_3b_temp=0.2_embd.pt" , type=str)
    parser.add_argument("--embd_path", default="/weka/home-reshinth/work/code-gen-div/embd_path/stripped_prompt") 

    args = parser.parse_args()
    main(args)
        

    


    

