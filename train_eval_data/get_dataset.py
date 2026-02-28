import os

import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import mode
from sklearn.model_selection import train_test_split

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Heterophilic dataset constants
# ---------------------------------------------------------------------------

HETEROPHILIC_DATASETS = [
    'chameleon', 'squirrel', 'actor', 'cornell', 'texas', 'wisconsin'
]

# Approximate edge homophily ratios (h = |{(u,v)∈E : y_u=y_v}| / |E|)
HOMOPHILY_RATIOS = {
    'chameleon':  0.23,
    'squirrel':   0.22,
    'actor':      0.22,
    'cornell':    0.20,
    'texas':      0.11,
    'wisconsin':  0.21,
    'cora':       0.81,
    'citeseer':   0.74,
}


def get_dataset(dataset_name: str):
    if dataset_name in ("citeseer", "cora"):
        try:
            return _load_npz(
                os.path.join(
                    os.path.dirname(__file__), "..", "data", dataset_name + ".npz"
                )
            )
        except FileNotFoundError as e:
            # Fallback for runs via SEML on the GPU cluster.
            raise e
            return _load_npz(f"{FALLBACK_SRC_PATH}/data/{dataset_name}.npz")
    elif dataset_name in ['flickr', 'reddit','dblp','pubmed', 'polblogs','acm','BlogCatalog','uai']:
        A = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", dataset_name, "adj.pt"))
        X = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", dataset_name, "fea.pt")).to(torch.float32)
        y = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", dataset_name, "label.pt"))
        
        return A, X, y
    elif dataset_name in HETEROPHILIC_DATASETS:
        # Try loading cached .pt files first; if not present, download & convert.
        return _load_or_download_heterophilic(dataset_name)
    else:
        A = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", "heter_data", dataset_name, "adj.pt"))
        X = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", "heter_data", dataset_name, "fea.pt")).to(torch.float32)
        y = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", "heter_data", dataset_name, "label.pt"))
        
        return A, X, y


# ---------------------------------------------------------------------------
# Heterophilic dataset utilities
# ---------------------------------------------------------------------------

def get_homophily_ratio(A: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute edge homophily ratio: fraction of edges connecting same-class nodes.

    h = |{(u,v) ∈ E : y_u = y_v}| / |E|

    h = 1.0 → perfectly homophilic (all edges within same class)
    h = 0.0 → perfectly heterophilic (all edges between different classes)

    Reference: Zhu et al. "Beyond Homophily in Graph Neural Networks" (2020)

    Args:
        A: [N, N] dense adjacency matrix (float)
        y: [N] integer label tensor

    Returns:
        homophily: float in [0, 1]
    """
    src, dst = A.nonzero(as_tuple=True)
    same_class = (y[src] == y[dst]).float()
    return same_class.mean().item()


def _load_or_download_heterophilic(
    name: str,
    root: str = None,
) -> tuple:
    """
    Load a heterophilic dataset, downloading it via torch_geometric if needed.

    Supported datasets:
        chameleon  (WikipediaNetwork, h ≈ 0.23)
        squirrel   (WikipediaNetwork, h ≈ 0.22)
        actor      (Actor/Film,       h ≈ 0.22)
        cornell    (WebKB,            h ≈ 0.20)
        texas      (WebKB,            h ≈ 0.11)
        wisconsin  (WebKB,            h ≈ 0.21)

    The data are cached as `data/heter_data/{name}/{adj,fea,label}.pt` in the
    same format used by the rest of the codebase, so subsequent loads are fast.

    Args:
        name: Dataset name (lowercase)
        root: Root dir for torch_geometric downloads (default: data/heter_data)

    Returns:
        (A, X, y) — dense float32 adjacency, float32 features, int64 labels
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "heter_data")
    cache_dir = os.path.join(base_dir, name)
    adj_path   = os.path.join(cache_dir, "adj.pt")
    fea_path   = os.path.join(cache_dir, "fea.pt")
    label_path = os.path.join(cache_dir, "label.pt")

    # ---- Fast path: load from cache ----------------------------------------
    if os.path.exists(adj_path) and os.path.exists(fea_path) and os.path.exists(label_path):
        A = torch.load(adj_path)
        X = torch.load(fea_path).to(torch.float32)
        y = torch.load(label_path)
        return A, X, y

    # ---- Slow path: download via torch_geometric and convert ----------------
    print(f"[get_dataset] Downloading heterophilic dataset '{name}' via torch_geometric ...")
    try:
        import torch_geometric.transforms as T
        from torch_geometric.utils import to_dense_adj
    except ImportError as e:
        raise ImportError(
            "torch_geometric is required for heterophilic datasets. "
            "Install it with:  pip install torch_geometric"
        ) from e

    download_root = root if root is not None else os.path.join(base_dir, "_pyg_downloads")
    transform = T.NormalizeFeatures()

    if name in ('chameleon', 'squirrel'):
        from torch_geometric.datasets import WikipediaNetwork
        dataset = WikipediaNetwork(root=download_root, name=name, transform=transform)
    elif name == 'actor':
        from torch_geometric.datasets import Actor
        dataset = Actor(root=download_root, transform=transform)
    elif name in ('cornell', 'texas', 'wisconsin'):
        from torch_geometric.datasets import WebKB
        dataset = WebKB(root=download_root, name=name, transform=transform)
    else:
        raise ValueError(f"Unknown heterophilic dataset: '{name}'")

    data = dataset[0]
    N = data.num_nodes

    # Convert to dense adjacency (codebase uses dense tensors)
    # to_dense_adj returns [1, N, N]; squeeze to [N, N]
    A = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0)
    # Ensure symmetry and no self-loops (matching _fix_adj_mat convention)
    A = (A + A.t()).clamp(max=1.0)
    A.fill_diagonal_(0.0)

    X = data.x.to(torch.float32)            # [N, F]
    y = data.y.to(torch.int64)              # [N]

    homophily = get_homophily_ratio(A, y)
    print(f"  Nodes: {N}, Edges: {int(A.sum().item()//2)}, "
          f"Features: {X.shape[1]}, Classes: {int(y.max().item())+1}, "
          f"Homophily: {homophily:.4f}")

    # Cache to disk
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(A, adj_path)
    torch.save(X, fea_path)
    torch.save(y, label_path)
    print(f"  Cached to {cache_dir}")

    return A, X, y


def load_heterophilic_dataset(name: str, root: str = None, split_seed: int = 42):
    """
    Public API: load a heterophilic dataset and return data with homophily info.

    Args:
        name:       Dataset name (lowercase), see HETEROPHILIC_DATASETS
        root:       Optional custom root for torch_geometric downloads
        split_seed: Random seed (splits are generated by get_splits() separately)

    Returns:
        A:          [N, N] dense float32 adjacency matrix
        X:          [N, F] float32 feature matrix
        y:          [N] int64 label tensor
        homophily:  float, measured edge homophily ratio
    """
    if name not in HETEROPHILIC_DATASETS:
        raise ValueError(
            f"Unknown heterophilic dataset: '{name}'. "
            f"Choose from: {HETEROPHILIC_DATASETS}"
        )
    A, X, y = _load_or_download_heterophilic(name, root=root)
    homophily = get_homophily_ratio(A, y)
    return A, X, y, homophily



def _load_npz(path: str):
    with np.load(path, allow_pickle=True) as loader:
        loader = dict(loader)
        A = _fix_adj_mat(_extract_csr(loader, "adj"))
        _, comp_ids = connected_components(A)
        lcc_nodes = np.nonzero(comp_ids == mode(comp_ids)[0])[0]
        A = torch.tensor(A[lcc_nodes, :][:, lcc_nodes].todense(), dtype=torch.float32)
        if "attr_data" in loader:
            X = torch.tensor(
                _extract_csr(loader, "attr")[lcc_nodes, :].todense(),
                dtype=torch.float32,
            )
        else:
            X = torch.eye(A.shape[0])
        if "labels" in loader:
            y = torch.tensor(loader["labels"][lcc_nodes], dtype=torch.int64)
        else:
            y = None
        return A, X, y


def _extract_csr(loader, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            loader[f"{prefix}_data"],
            loader[f"{prefix}_indices"],
            loader[f"{prefix}_indptr"],
        ),
        loader[f"{prefix}_shape"],
    )


def _fix_adj_mat(A: sp.csr_matrix) -> sp.csr_matrix:
    # Some adjacency matrices do have some values on the diagonal, but not everywhere. Get rid of this mess.
    A = A - sp.diags(A.diagonal())
    # For some reason, some adjacency matrices are not symmetric. Fix this following the Nettack code.
    A = A + A.T
    A[A > 1] = 1
    return A


def get_splits(
    y, 
    more_sps=0,
):
    """
    Produces 5 deterministic 10-10-80 splits.
    """
    if more_sps != 0:
        return [
            _three_split(y.cpu(), 0.1, 0.1, random_state=r)
            for r in [1234, 2021, 1309, 4242, 1698] + list(range(more_sps))
        ]
    return [
        _three_split(y.cpu(), 0.1, 0.1, random_state=r)
        #for r in [1234, 2021, 1309, 4242, 1698]
        for r in [1534, 2021, 1323, 1535, 1698]
    ]
    

def _three_split(
    y, size_1, size_2, random_state
):
    idx = np.arange(y.shape[0])
    idx_12, idx_3 = train_test_split(
        idx, train_size=size_1 + size_2, stratify=y, random_state=random_state
    )
    idx_1, idx_2 = train_test_split(
        idx_12,
        train_size=size_1 / (size_1 + size_2),
        stratify=y[idx_12],
        random_state=random_state,
    )
    return torch.tensor(idx_1), torch.tensor(idx_2), torch.tensor(idx_3)
