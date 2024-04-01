import torch


def get_centroid(data):
    """
    Calculates the centroid of a cluster.
    """
    return data.mean(dim=0)

def sum_squared_errors(data, centroid):
    """
    Calculates the sum of squared errors (SSE) within a cluster.
    """
    distances = torch.sum((data - centroid) ** 2, dim=1)
    return torch.sum(distances)

def average_pairwise_distance(data):
    """
    Calculates the average pairwise distance within a cluster.
    """
    pairwise_distances = torch.cdist(data, data)
    upper_triangular = pairwise_distances.triu(diagonal=1)
    return torch.mean(upper_triangular[upper_triangular != 0])

def silhouette_coefficient(data, centroid, cluster_labels):
    """
    Calculates the average silhouette coefficient within a cluster.
    """
    distances = torch.sum((data - centroid) ** 2, dim=1)
    a = torch.mean(distances)

    other_clusters = torch.unique(cluster_labels)
    if len(other_clusters) == 1:
        # If all data points belong to the same cluster, set the silhouette coefficient to a predefined value
        return 1.0  # or 0.0, depending on your preference

    other_clusters = other_clusters[other_clusters != cluster_labels]
    b = torch.zeros(data.shape[0])

    for other_cluster in other_clusters:
        other_centroid = data[cluster_labels == other_cluster].mean(dim=0)
        b_temp = torch.sum((data - other_centroid) ** 2, dim=1)
        b = torch.minimum(b, torch.mean(b_temp))

    s = (b - a) / torch.maximum(a, b)
    return torch.mean(s)

if __name__ == "__main__":
    data = torch.randn(100, 5)  
    centroid = get_centroid(data)
    # Calculate metrics
    sse = sum_squared_errors(data, centroid)
    avg_pairwise_dist = average_pairwise_distance(data)
    sil_coeff = silhouette_coefficient(data, centroid, torch.zeros(100))  # Assuming all data points belong to the same cluster

    print(f"Sum of Squared Errors: {sse}")
    print(f"Average Pairwise Distance: {avg_pairwise_dist}")
    print(f"Silhouette Coefficient: {sil_coeff}")