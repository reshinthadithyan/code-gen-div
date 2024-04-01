import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse




def load_torch_tensor(path):
    loaded_dict = torch.load(path)
    return loaded_dict

def create_labels(tensor_dict):
    x = []
    labels = []
    for key in tensor_dict:
        for i in range(tensor_dict[key].shape[0]):
            x.append(tensor_dict[key][i].numpy())
            labels.append(key)

    x = np.asarray(x)
    print(x.shape)
    return x, labels



def process_and_plot_pca(tensor_path):
    # Load the torch tensor
    tensor = load_torch_tensor(tensor_path)
    # Create labels
    x, labels = create_labels(tensor)
    # PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(x)
    reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    #Add labels to the reduced_df
    reduced_df['labels'] = labels
    # Plotting
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue=labels, palette='tab10',legend=None, alpha=0.5)
    #Add title
    plt.title(f'PCA projection of {tensor_path.split("/")[-1]}')        
    plot_save_path = tensor_path.split("/")[-1].replace(".pt","") + "_pca_plot.png"
    plt.savefig("plots/"+plot_save_path)

if __name__ == "__main__":
    # Load the torch tensor
    tensor_path = "/weka/home-reshinth/work/code-gen-div/embd_path/ds_1.3B_temp=0.2_embd.pt"   
    process_and_plot_pca(tensor_path)