import torch

def build_syn_graph(heads, seq_len=128, total_nodes=162):
    edges = []
    for i, h_idx in enumerate(heads):
        edges.append([i, i])
        if 0 <= h_idx < seq_len:
            edges.append([i, h_idx])
            edges.append([h_idx, i])
            
    for i in range(seq_len, total_nodes):
        edges.append([i, i])
        
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def build_sem_graph(seq_len=128, total_nodes=162, window=2):
    edges = []
    for i in range(seq_len):
        start = max(0, i - window)
        end = min(seq_len, i + window + 1)
        for j in range(start, end):
            edges.append([i, j])
            
    for i in range(seq_len, total_nodes):
        edges.append([i, i])
        
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def build_asp_graph(seq_len=128, num_aspects=34):
    edges = []
    total_nodes = seq_len + num_aspects
    
    for i in range(total_nodes):
        edges.append([i, i])
        
    aspect_indices = range(seq_len, total_nodes) 
    
    for asp_idx in aspect_indices:
        for i in range(seq_len):
            edges.append([asp_idx, i])
            edges.append([i, asp_idx])
            
    return torch.tensor(edges, dtype=torch.long).t().contiguous()