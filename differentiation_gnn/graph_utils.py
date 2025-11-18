import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import numpy as np

def create_spatiotemporal_graph(adj_list, img_data, region, node_features_list=None):
    """
    Create a spatiotemporal graph from a list of adjacency dictionaries.

    Args:
        adj_list (list of dict): [adj_dict_t0, adj_dict_t1, ...], with key being a node_id and value being a list of
        neighboring node_ids
        img_data (MultiIndex dataframe): contains feature data for each cell
        node_features_list (list of torch.Tensor, optional): Features for each timepoint [x_t0, x_t1, ...]

    Returns:
        Data: PyTorch Geometric Data object
    """

    edge_index = []
    node_ids_all = []
    time_ids_all = []
    trackIDs = []

    # Split dataset by region
    all_df = pd.read_pickle(img_data)
    all_df['Region','Meta'] = [f[0] for f in 
                            all_df.index.get_level_values(1).str.split('_',expand=True)]
    logical_index = all_df['Region','Meta'] == region
    region_df = all_df[logical_index]

    # Make a list of lists containing trackIDs per frame
    all_trackIDs = [int(f[1]) for f in 
                    region_df.index.get_level_values(1).str.split('_',expand=True)]
    num_frames = min(len(adj_list), np.max(region_df.index.get_level_values(0)))
    num_cells_in_frame = region_df.index.get_level_values(0).value_counts()
    offset = 0
    for frame in range(0, num_frames):
        frameIDs = all_trackIDs[offset: (offset + num_cells_in_frame[frame] - 1)]
        offset += num_cells_in_frame[frame]
        trackIDs.append(frameIDs)

    
    # Extract and save inheritance data
    inheritance = all_df['Mother','Meta']
    non_nan_indices = inheritance[inheritance.notna()].index.tolist()
    inheritance_non_nan = inheritance[non_nan_indices]
    inherited_ids = [int(f[1]) for f in 
                    inheritance_non_nan.index.get_level_values(1).str.split('_',expand=True)]

    num_nodes = max(max(adj.keys()) for adj in adj_list) + 1  # assume node IDs are integers starting at 0

    # --- Build spatial edges ---
    for t, adj in enumerate(adj_list):
        offset = t * num_nodes  # simpler: just offset by timepoint index * max nodes
        for src, neighbors in adj.items():
            for dst in neighbors:
                edge_index.append([offset + src, offset + dst])
        node_ids_all.extend(adj.keys())
        time_ids_all.extend([t] * len(adj))

    # --- Build temporal edges ---
    for t in range(num_frames - 1):
        adj_curr = adj_list[t]
        adj_next = adj_list[t + 1]
        offset_curr = t * num_nodes
        offset_next = (t + 1) * num_nodes

        shared_nodes = set(adj_curr.keys()).intersection(adj_next.keys())
        for nid in shared_nodes:
            edge_index.append([offset_curr + nid, offset_next + nid])
            edge_index.append([offset_next + nid, offset_curr + nid])  # bidirectional
        
        mother_nodes = sorted(map(int, set(trackIDs[t]).intersection(inheritance_non_nan[t+1])))
        for node in mother_nodes:
            nid = [index for index, value in enumerate(sorted(inheritance_non_nan[t+1])) if value == node]
            daughter_nodes = [inherited_ids[i] for i in nid]
            for daughter in daughter_nodes:
                edge_index.append([offset_curr + node, offset_next + daughter])

    # Convert edge list to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # --- Node features ---
    if node_features_list is not None:
        x = torch.cat(node_features_list, dim=0)
    else:
        # Default: dummy features
        total_nodes = num_nodes * num_frames
        x = torch.arange(total_nodes, dtype=torch.float).unsqueeze(1)

    # --- Node IDs and time IDs ---
    node_ids = torch.tensor(node_ids_all, dtype=torch.long)
    time_ids = torch.tensor(time_ids_all, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, node_ids=node_ids, time_ids=time_ids)

    return data