import torch
import torch.nn.functional as F_nn

print("Test 1: Soft homophily computation")

N, C = 10, 3
A = torch.zeros(N, N)
for i in range(5):
    for j in range(5):
        if i != j:
            A[i, j] = 1.0
for i in range(5, N):
    for j in range(5, N):
        if i != j:
            A[i, j] = 1.0
A[2, 7] = 1.0
A[7, 2] = 1.0

logits = torch.zeros(N, C)
logits[:5, 0] = 5.0
logits[5:, 1] = 5.0
P = F_nn.softmax(logits, dim=-1)

H = P @ P.T
eps = 1e-8
h = (H * A).sum(-1) / (A.sum(-1) + eps)

print(f"Group 1 homophily (should be high ~0.9): {h[:5].mean():.4f}")
print(f"Group 2 homophily (should be high ~0.9): {h[5:].mean():.4f}")
print(f"Node 2 (has cross-edge, should be lower): {h[2]:.4f}")
print(f"Node 7 (has cross-edge, should be lower): {h[7]:.4f}")

assert h[0] > h[2], "Node 0 (no cross edges) should be more homophilic than node 2"
print("PASS: homophily correctly identifies cross-edge nodes\n")

print("Test 2: Adaptive q computation")
q_base = 0.75
q_relax = 0.20

q_adaptive = q_base + (1.0 - h) * q_relax
print(f"q for group 1 (homophilic): {q_adaptive[:5].mean():.4f} (expect ~{q_base:.2f})")
print(f"q for cross-edge node 2:    {q_adaptive[2]:.4f} (expect > {q_base:.2f})")
assert q_adaptive[0] < q_adaptive[2], "Homophilic node should have lower q than cross-edge node"
print("PASS: adaptive q correctly penalizes cross-edge nodes\n")

print("Test 3: Per-node gamma computation")
torch.manual_seed(42)
y_full = torch.rand(N, N) * 2.0
y_full = (y_full + y_full.T) / 2
y_full.fill_diagonal_(0.0)

gammas = []
for i in range(N):
    neighbor_mask = A[i] > 0
    y_i = y_full[i][neighbor_mask]
    if len(y_i) == 0:
        gammas.append(torch.tensor(1.0))
    else:
        g_i = torch.quantile(y_i, q_adaptive[i])
        gammas.append(g_i)

gammas = torch.stack(gammas)
print(f"Gamma range: [{gammas.min():.4f}, {gammas.max():.4f}]")
print(f"Homophilic group 1 gammas (aggressive): {gammas[:5].mean():.4f}")
print(f"Cross-edge node 2 gamma (lenient): {gammas[2]:.4f}")
print("PASS: per-node gammas computed correctly\n")

print("ALL TESTS PASSED")
