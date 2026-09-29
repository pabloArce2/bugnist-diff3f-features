"""Point sampling and chunked cosine nearest-neighbour search between descriptor sets."""

import numpy as np
import torch
from tqdm import tqdm


def farthest_point_indices(vertices, count, seed):
    """Farthest-point sampling, starting from the vertex farthest from the centroid.

    Every 10th pick is drawn at random from the 64 farthest candidates, so that
    the samples do not all end up on one long appendage.
    """
    rng = np.random.default_rng(seed)
    count = min(count, len(vertices))
    center = vertices.mean(axis=0, keepdims=True)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.argmax(np.sum((vertices - center) ** 2, axis=1)))
    min_dist2 = np.sum((vertices - vertices[selected[0]]) ** 2, axis=1)

    for i in range(1, count):
        if i % 10 == 0:
            top_count = min(64, len(min_dist2))
            candidates = np.argpartition(min_dist2, -top_count)[-top_count:]
            chosen = int(rng.choice(candidates))
        else:
            chosen = int(np.argmax(min_dist2))
        selected[i] = chosen
        min_dist2 = np.minimum(min_dist2, np.sum((vertices - vertices[chosen]) ** 2, axis=1))
    return selected


def sample_indices(vertices, count, sampling, seed):
    if sampling == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(len(vertices), size=min(count, len(vertices)), replace=False)
    return farthest_point_indices(vertices, count, seed)


def nearest_neighbors(query, target, device, query_chunk_size=8, target_chunk_size=8192):
    """Index and cosine score of the most similar target row for every query row."""
    query = torch.nn.functional.normalize(query, dim=1)
    target = torch.nn.functional.normalize(target, dim=1)
    best_indices = torch.empty(len(query), dtype=torch.long)
    best_scores = torch.full((len(query),), -float("inf"), dtype=torch.float32)

    for q_start in tqdm(range(0, len(query), query_chunk_size), desc="Query chunks"):
        q_stop = min(q_start + query_chunk_size, len(query))
        rows = query[q_start:q_stop].to(device)
        chunk_scores = torch.full((q_stop - q_start,), -float("inf"), device=device)
        chunk_indices = torch.zeros((q_stop - q_start,), dtype=torch.long, device=device)

        for t_start in range(0, len(target), target_chunk_size):
            block = target[t_start : t_start + target_chunk_size].to(device)
            scores, local = (rows @ block.T).max(dim=1)
            update = scores > chunk_scores
            chunk_scores[update] = scores[update]
            chunk_indices[update] = local[update] + t_start

        best_indices[q_start:q_stop] = chunk_indices.cpu()
        best_scores[q_start:q_stop] = chunk_scores.cpu()

    return best_indices.numpy(), best_scores.numpy()


def nearest_two(query, target, device, query_chunk_size=16, target_chunk_size=8192):
    """Like nearest_neighbors, but also returns the runner-up score (for the NN margin)."""
    query = torch.nn.functional.normalize(query, dim=1)
    target = torch.nn.functional.normalize(target, dim=1)
    best_indices = torch.empty(len(query), dtype=torch.long)
    best_scores = torch.full((len(query),), -float("inf"), dtype=torch.float32)
    second_scores = torch.full((len(query),), -float("inf"), dtype=torch.float32)

    for q_start in tqdm(range(0, len(query), query_chunk_size), desc="Query chunks"):
        q_stop = min(q_start + query_chunk_size, len(query))
        rows = query[q_start:q_stop].to(device)
        best = torch.full((q_stop - q_start,), -float("inf"), device=device)
        second = torch.full((q_stop - q_start,), -float("inf"), device=device)
        best_index = torch.zeros((q_stop - q_start,), dtype=torch.long, device=device)

        for t_start in range(0, len(target), target_chunk_size):
            block = target[t_start : t_start + target_chunk_size].to(device)
            similarity = rows @ block.T
            values, local = torch.topk(similarity, k=min(2, similarity.shape[1]), dim=1)
            top = values[:, 0]
            runner_up = values[:, 1] if values.shape[1] > 1 else torch.full_like(top, -float("inf"))

            update = top > best
            second = torch.where(update, torch.maximum(best, runner_up), second)
            second = torch.where(~update, torch.maximum(second, top), second)
            best = torch.where(update, top, best)
            best_index = torch.where(update, local[:, 0] + t_start, best_index)

        best_indices[q_start:q_stop] = best_index.cpu()
        best_scores[q_start:q_stop] = best.cpu()
        second_scores[q_start:q_stop] = second.cpu()

    return best_indices.numpy(), best_scores.numpy(), second_scores.numpy()
