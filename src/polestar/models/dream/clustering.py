import torch
import torch.nn.functional as F
def k_means(x, num_clusters, num_iters=5, distance="cosine"):
    B, H, L, D = x.shape
    device = x.device
    dtype = x.dtype

    if L <= num_clusters:
        if L < num_clusters:
            repeats = num_clusters // L + 1
            centroids = x.repeat(1, 1, repeats, 1)[:, :, :num_clusters, :]
        else:
            centroids = x.clone()
        labels = torch.arange(L, device=device).view(1, 1, -1).expand(B, H, -1)
        return centroids, labels

    if distance == "cosine":
        x_used = F.normalize(x, p=2, dim=-1)
    elif distance == "l2":
        x_used = x
    else:
        raise ValueError(f"Unsupported distance: {distance}")

    rand_idx = torch.randperm(L, device=device)[:num_clusters]
    centroids = x[:, :, rand_idx, :].clone()

    for _ in range(num_iters):
        if distance == "cosine":

            centroids_norm =  F.normalize(centroids, p=2, dim=-1)
            sim = torch.matmul(x_used, centroids_norm.transpose(-2, -1))
            labels = sim.argmax(dim=-1)
        else:

            x2 = (x_used * x_used).sum(dim=-1, keepdim=True)
            c2 = (centroids * centroids).sum(dim=-1).unsqueeze(-2)
            xc = torch.matmul(x_used, centroids.transpose(-2, -1))
            dist2 = x2 + c2 - 2.0 * xc
            labels = dist2.argmin(dim=-1)

        mask = F.one_hot(labels, num_clusters).to(dtype=dtype)

        sum_features = torch.matmul(mask.transpose(-2, -1), x)
        cluster_counts = mask.sum(dim=2).unsqueeze(-1)
        new_centroids = sum_features / (cluster_counts + 1e-6)

        is_empty = (cluster_counts == 0)
        if is_empty.any():
            random_indices = torch.randint(0, L, (num_clusters,), device=device)
            random_points = x[:, :, random_indices, :]
            new_centroids = torch.where(is_empty, random_points, new_centroids)

        centroids = new_centroids

    return centroids.to(dtype), labels

def cluster_past_hidden_states(past_hidden_states, num_clusters=64):
    """
    Wrapper for Hidden States Clustering.
    Args:
        past_hidden_states: List[Tensor], each (Batch, Seq_Len, Hidden_Dim)
    Outputs:
        List of Tuples: (centroids, labels)
            centroids: (Batch, K, Hidden_Dim) - without the head dimension
            labels: (Batch, Seq_Len) - without the head dimension
    """
    results = []

    for layer_idx, hidden_state in enumerate(past_hidden_states):

        if hidden_state.dim() == 3:
            hidden_state_4d = hidden_state.unsqueeze(1)

            with torch.no_grad():
                centroids, labels = k_means(hidden_state_4d, num_clusters)

            results.append((centroids.squeeze(1), labels.squeeze(1)))

        else:
            with torch.no_grad():
                centroids, labels = k_means(hidden_state, num_clusters)

            results.append((centroids, labels))

    return results
