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
    'pubmed':     0.80,
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
    elif dataset_name == 'pubmed':
        # Try loading from cache first, then download via Planetoid if needed
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "pubmed")
        adj_path = os.path.join(cache_dir, "adj.pt")
        fea_path = os.path.join(cache_dir, "fea.pt")
        label_path = os.path.join(cache_dir, "label.pt")
        
        # Try loading from cache
        if os.path.exists(adj_path) and os.path.exists(fea_path) and os.path.exists(label_path):
            A = torch.load(adj_path)
            X = torch.load(fea_path).to(torch.float32)
            y = torch.load(label_path)
            return A, X, y
        
        # Download via Planetoid if not cached
        return _load_or_download_pubmed()
    elif dataset_name in ['flickr', 'reddit','dblp', 'polblogs','acm','BlogCatalog','uai']:
        A = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", dataset_name, "adj.pt"))
        X = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", dataset_name, "fea.pt")).to(torch.float32)
        y = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", dataset_name, "label.pt"))
        
        return A, X, y
    elif dataset_name.startswith('ogbn-'):
        # Load Open Graph Benchmark datasets (ogbn-arxiv, ogbn-products, etc.)
        return _load_or_download_ogb(dataset_name)
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

def _load_pubmed_direct(base_dir: str) -> tuple:
    """
    Download PubMed data DIRECTLY without torch_geometric (avoids JIT compilation issues).
    
    Uses the canonical Planetoid repository on GitHub.
    Parses the pickle files and constructs edge index manually.
    
    Returns:
        (A, X, y) — dense float32 adjacency, float32 features, int64 labels
    """
    import pickle
    import urllib.request
    from pathlib import Path
    
    os.makedirs(base_dir, exist_ok=True)
    base_url = "https://github.com/kimiyoung/planetoid/raw/master/data/"
    
    # Files to download
    files_to_download = {
        'graph': 'ind.pubmed.graph',
        'x': 'ind.pubmed.x',
        'y': 'ind.pubmed.y',
        'tx': 'ind.pubmed.tx',
        'ty': 'ind.pubmed.ty',
        'test_idx': 'ind.pubmed.test.index',
    }
    
    local_files = {}
    print(f"  [direct download] Downloading PubMed data from GitHub...")
    
    for key, filename in files_to_download.items():
        local_path = os.path.join(base_dir, filename)
        if not os.path.exists(local_path):
            url = base_url + filename
            print(f"    Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download {filename} from {url}: {e}") from e
        local_files[key] = local_path
    
    # Parse files
    def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    
    print(f"  [direct download] Parsing data...")
    x = load_pickle(local_files['x'])  # [train_size, features]
    y = load_pickle(local_files['y'])  # [train_size, num_classes] (one-hot)
    tx = load_pickle(local_files['tx'])  # [test_size, features]
    ty = load_pickle(local_files['ty'])  # [test_size, num_classes] (one-hot)
    graph = load_pickle(local_files['graph'])  # {node_id: [neighbor_ids]}
    
    # Load test indices
    test_idx = np.loadtxt(local_files['test_idx'], dtype=np.int32)
    
    # Convert sparse matrices to dense if needed
    if sp.issparse(x):
        x = x.toarray()
    if sp.issparse(tx):
        tx = tx.toarray()
    
    # Combine train and test features
    X = np.vstack([x, tx]).astype(np.float32)
    
    # Combine train and test labels (convert from one-hot)
    y_train = np.argmax(y, axis=1)
    y_test = np.argmax(ty, axis=1)
    y_full = np.zeros(X.shape[0], dtype=np.int32)
    y_full[: len(y_train)] = y_train
    y_full[len(y_train):] = y_test
    
    # Build adjacency from graph dict
    N = X.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    for src_node, dst_neighbors in graph.items():
        for dst_node in dst_neighbors:
            if src_node < N and dst_node < N:
                A[src_node, dst_node] = 1.0
                A[dst_node, src_node] = 1.0  # Undirected
    
    # Remove self-loops
    np.fill_diagonal(A, 0.0)
    
    # Convert to torch tensors
    A = torch.from_numpy(A).to(torch.float32)
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y_full).to(torch.int64)
    
    # Verify dimensions
    num_classes = int(y.max().item()) + 1
    print(f"  [direct download] Loaded: {N} nodes, {int(A.sum().item()//2)} edges, "
          f"{X.shape[1]} features, {num_classes} classes")
    
    return A, X, y


def _load_or_download_pubmed(
    root: str = None,
) -> tuple:
    """
    Load PubMed dataset. First tries direct download (avoiding torch_geometric JIT issues).
    Falls back to torch_geometric if direct download fails.
    
    Args:
        root: Optional root for downloads (default: data/pubmed)
    
    Returns:
        (A, X, y) — dense float32 adjacency, float32 features, int64 labels
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "pubmed")
    adj_path = os.path.join(base_dir, "adj.pt")
    fea_path = os.path.join(base_dir, "fea.pt")
    label_path = os.path.join(base_dir, "label.pt")
    
    # Check cache first
    if os.path.exists(adj_path) and os.path.exists(fea_path) and os.path.exists(label_path):
        A = torch.load(adj_path)
        X = torch.load(fea_path).to(torch.float32)
        y = torch.load(label_path)
        return A, X, y
    
    print(f"[get_dataset] Downloading PubMed dataset...")
    
    # FIRST ATTEMPT: Direct download (avoids torch_geometric JIT issues)
    try:
        A, X, y = _load_pubmed_direct(base_dir)
        # Cache to disk
        torch.save(A, adj_path)
        torch.save(X, fea_path)
        torch.save(y, label_path)
        print(f"  Cached to {base_dir}")
        return A, X, y
    except Exception as e:
        print(f"  [direct download] Failed: {e}")
        print(f"  [torch_geometric fallback] Attempting torch_geometric...")
    
    # FALLBACK: torch_geometric (if direct download fails)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            from torch_geometric.datasets import Planetoid
            from torch_geometric.utils import to_dense_adj
    except (ImportError, RuntimeError, OSError) as e:
        raise ImportError(
            f"PubMed download failed. Both direct and torch_geometric methods failed.\n"
            f"Try installing torch_geometric: pip install torch_geometric\n"
            f"Error: {e}"
        ) from e
    
    download_root = root if root is not None else base_dir
    try:
        dataset = Planetoid(root=download_root, name='pubmed')
        data = dataset[0]
        N = data.num_nodes
        
        # Convert to dense adjacency
        A = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0)
        A = (A + A.t()).clamp(max=1.0)
        A.fill_diagonal_(0.0)
        
        X = data.x.to(torch.float32)
        y = data.y.to(torch.int64)
        
        num_classes = int(y.max().item()) + 1
        print(f"  Nodes: {N}, Edges: {int(A.sum().item()//2)}, "
              f"Features: {X.shape[1]}, Classes: {num_classes}")
        
        # Cache to disk
        torch.save(A, adj_path)
        torch.save(X, fea_path)
        torch.save(y, label_path)
        print(f"  Cached to {base_dir}")
    except (RuntimeError, OSError) as e:
        raise RuntimeError(
            f"Failed to download PubMed dataset via torch_geometric. \n"
            f"Try upgrading: pip install --upgrade torch_geometric\n"
            f"Error: {e}"
        ) from e
    
    return A, X, y


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
        # Defer import and suppress torch_geometric JIT issues
        import warnings
        import os as os_module
        
        # Set environment to suppress JIT compilation
        os_module.environ['TORCH_JIT_IGNORE_LCHECK'] = '1'
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            # Import at function level to avoid module-load JIT errors
            import torch_geometric.transforms as T
            from torch_geometric.utils import to_dense_adj
    except (ImportError, RuntimeError, OSError) as e:
        raise ImportError(
            f"torch_geometric is required for heterophilic datasets ('{name}'). \n"
            f"Install it with:  pip install torch_geometric\n"
            f"Or try downgrading: pip install 'torch_geometric==2.3.0'\n"
            f"Error: {e}"
        ) from e

    download_root = root if root is not None else os.path.join(base_dir, "_pyg_downloads")
    transform = T.NormalizeFeatures()

    try:
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
    except (RuntimeError, OSError, TypeError) as e:
        raise RuntimeError(
            f"Failed to download heterophilic dataset '{name}'. \n"
            f"This may be due to torch_geometric version incompatibility. \n"
            f"Try upgrading: pip install --upgrade torch_geometric\n"
            f"Error: {e}"
        ) from e

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




def _load_ogbn_arxiv_direct(base_dir: str) -> tuple:
    """
    Download ogbn-arxiv data DIRECTLY without OGB library problems.
    Manually downloads and parses arxiv dataset.
    Returns:
        (A, X, y) — dense float32 adjacency, float32 features, int64 labels
    """
    import pickle
    import urllib.request
    import zipfile
    
    os.makedirs(base_dir, exist_ok=True)
    
    # Check cache first
    adj_path = os.path.join(base_dir, "adj.pt")
    fea_path = os.path.join(base_dir, "fea.pt")
    label_path = os.path.join(base_dir, "label.pt")
    
    if os.path.exists(adj_path) and os.path.exists(fea_path) and os.path.exists(label_path):
        print(f"  [direct] Loading cached ogbn-arxiv")
        A = torch.load(adj_path)
        X = torch.load(fea_path).to(torch.float32)
        y = torch.load(label_path)
        return A, X, y
    
    print(f"  [direct] Downloading ogbn-arxiv from Stanford...")
    
    # Download zip file
    base_url = "http://snap.stanford.edu/ogb/data/nodeproppred/"
    zip_filename = "ogbn_arxiv.zip"
    zip_path = os.path.join(base_dir, zip_filename)
    
    if not os.path.exists(zip_path):
        url = base_url + zip_filename
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"    Downloaded successfully ({os.path.getsize(zip_path) / 1024 / 1024:.1f} MB)")
        except Exception as e:
            raise RuntimeError(f"Failed to download from {url}: {e}") from e
    
    # Extract zip file
    print(f"  [direct] Extracting data...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(base_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract {zip_path}: {e}") from e
    
    # Find extracted directory
    arxiv_dir = os.path.join(base_dir, "ogbn_arxiv", "raw")
    if not os.path.exists(arxiv_dir):
        arxiv_dir = os.path.join(base_dir, "raw")
    
    # Load graph data
    graph_path = os.path.join(arxiv_dir, "data.pkl")
    if not os.path.exists(graph_path):
        graph_path = os.path.join(arxiv_dir,  "graph.pkl")
    
    print(f"  [direct] Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f, encoding='latin1')
    
    if not isinstance(graph, dict) or 'node_feat' not in graph:
        raise RuntimeError(f"Unexpected graph format")
    
    X = graph['node_feat'].astype(np.float32)
    edge_index = graph['edge_index']
    N = X.shape[0]
    
    # Build adjacency
    A = np.zeros((N, N), dtype=np.float32)
    for i, j in zip(edge_index[0], edge_index[1]):
        if i < N and j < N:
            A[i, j] = 1.0
    np.fill_diagonal(A, 0.0)
    
    # Load labels
    label_path_src = os.path.join(arxiv_dir, "node-label.npy")
    if not os.path.exists(label_path_src):
        label_path_src = os.path.join(arxiv_dir, "label.npy")
    y = np.load(label_path_src).flatten().astype(np.int64)
    
    # Convert to torch
    X = torch.from_numpy(X).to(torch.float32)
    A = torch.from_numpy(A).to(torch.float32)
    y = torch.from_numpy(y).to(torch.int64)
    
    print(f"  [direct] Loaded: {N} nodes, {int(A.sum().item()//2)} edges, {X.shape[1]} features, {int(y.max())+1} classes")
    
    # Cache
    torch.save(A, adj_path)
    torch.save(X, fea_path)
    torch.save(y, label_path)
    print(f"  [direct] Cached to {base_dir}")
    
    return A, X, y


def _load_or_download_ogb(
    name: str,
    root: str = None,
) -> tuple:
    """
    Load an Open Graph Benchmark (OGB) dataset, downloading if needed.

    Supported datasets:
        ogbn-arxiv  (Citation network, ~169K nodes, ~1M edges)

    The data are cached as `data/ogb/{name}/{adj,fea,label}.pt` for fast re-loading.

    Args:
        name: Dataset name starting with 'ogbn-', e.g., 'ogbn-arxiv'
        root: Optional custom root for OGB downloads (default: data/ogb)

    Returns:
        (A, X, y) — dense float32 adjacency, float32 features, int64 labels
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "ogb")
    cache_dir = os.path.join(base_dir, name)
    adj_path   = os.path.join(cache_dir, "adj.pt")
    fea_path   = os.path.join(cache_dir, "fea.pt")
    label_path = os.path.join(cache_dir, "label.pt")

    # Fast path: load from cache
    if os.path.exists(adj_path) and os.path.exists(fea_path) and os.path.exists(label_path):
        print(f"[get_dataset] Loading cached OGB dataset '{name}'")
        A = torch.load(adj_path)
        X = torch.load(fea_path).to(torch.float32)
        y = torch.load(label_path)
        return A, X, y

    print(f"[get_dataset] Downloading OGB dataset '{name}'...")
    
    # Try direct download for ogbn-arxiv first
    if name == 'ogbn-arxiv':
        try:
            return _load_ogbn_arxiv_direct(cache_dir)
        except Exception as e:
            print(f"  Direct download failed: {e}")
            print(f"  Trying OGB library as fallback...")
    
    # Fallback: Use OGB library
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            from ogb.nodeproppred import NodePropPredDataset
            from torch_geometric.utils import to_dense_adj
    except (ImportError, RuntimeError, OSError) as e:
        raise ImportError(
            f"ogb and torch_geometric required. Install with: pip install ogb torch_geometric\n"
            f"Error: {e}"
        ) from e

    download_root = root if root is not None else os.path.join(base_dir, "_ogb_downloads")
    
    # Monkey-patch torch.load for PyTorch 2.6+
    _original_torch_load = torch.load
    def _patched_torch_load(f, *args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(f, *args, **kwargs)
    
    try:
        torch.load = _patched_torch_load
        
        dataset = NodePropPredDataset(name=name, root=download_root)
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        
        # Handle tuple returns
        if isinstance(data, tuple) and isinstance(data[0], dict) and 'num_nodes' in data[0]:
            graph_dict = data[0]
            N = graph_dict['num_nodes']
            edge_index = graph_dict['edge_index']
            X = torch.from_numpy(graph_dict['node_feat']).to(torch.float32)
            y = torch.from_numpy(data[1].flatten()).to(torch.int64) if len(data) > 1 else None
            A = torch.zeros((N, N), dtype=torch.float32)
            for i, j in zip(edge_index[0], edge_index[1]):
                A[i, j] = 1.0
            A.fill_diagonal_(0.0)
        else:
            N = data.num_nodes
            A = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0)
            A = (A + A.t()).clamp(max=1.0)
            A.fill_diagonal_(0.0)
            X = data.x.to(torch.float32)
            y = data.y.squeeze(-1).to(torch.int64)
        
        num_classes = int(y.max().item()) + 1
        print(f"  Nodes: {N}, Edges: {int(A.sum().item()//2)}, Features: {X.shape[1]}, Classes: {num_classes}")
        
        # Cache
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(A, adj_path)
        torch.save(X, fea_path)
        torch.save(y, label_path)
        print(f"  Cached to {cache_dir}")

        return A, X, y
    except (RuntimeError, OSError, TypeError, AttributeError) as e:
        raise RuntimeError(
            f"Failed to load OGB dataset '{name}': {e}"
        ) from e
    finally:
        torch.load = _original_torch_load


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
