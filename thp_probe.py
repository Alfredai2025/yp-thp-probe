#!/usr/bin/env python3
"""THP effect on graph-ANN walks (hnswlib, single thread, SYNTHETIC clustered data).
Self-contained: no downloads. GT computed in-runner (deterministic, SEED fixed).
Fingerprint = md5 of all result sets — must be bit-identical across THP arms.
"""
import hnswlib, numpy as np, hashlib, time

SEED = 42
rng = np.random.default_rng(SEED)
N, D, NQ = 200_000, 128, 1_000
centers = rng.normal(size=(2_000, D)).astype(np.float32)
centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
ci = rng.integers(0, len(centers), N)
XB = (centers[ci] + 0.05 * rng.normal(size=(N, D)).astype(np.float32)).astype(np.float32)
qi = rng.integers(0, len(centers), NQ)
XQ = (centers[qi] + 0.05 * rng.normal(size=(NQ, D)).astype(np.float32)).astype(np.float32)
gt = np.empty((NQ, 10), dtype=np.int64)
xb2 = (XB ** 2).sum(axis=1)
for a in range(0, NQ, 250):
    b = min(a + 250, NQ)
    d2 = (XQ[a:b] ** 2).sum(axis=1)[:, None] + xb2[None, :] - 2.0 * (XQ[a:b] @ XB.T)
    gt[a:b] = np.argpartition(d2, 10, axis=1)[:, :10]

M, EF_C, EF_S = 16, 200, 128
t0 = time.time()
index = hnswlib.Index(space='l2', dim=D)
index.init_index(max_elements=N, M=M, ef_construction=EF_C, random_seed=SEED)
index.add_items(XB, np.arange(N), num_threads=1)
index.set_ef(EF_S)
print(f"build {time.time()-t0:.0f}s", flush=True)

index.knn_query(XQ[:100], k=10, num_threads=1)   # warmup
reps = 5
lat = []
res_hash = hashlib.md5()
recalls = []
for r in range(reps):
    t0 = time.time()
    labels, _ = index.knn_query(XQ, k=10, num_threads=1)
    el = time.time() - t0
    lat.append(el)
    res_hash.update(np.ascontiguousarray(labels).tobytes())
    recalls.append(float(np.mean([len(set(labels[i]) & set(gt[i])) / 10 for i in range(NQ)])))
med = sorted(lat)[len(lat) // 2]
print(f"RESULT reps={reps} med_query_s={med:.4f} qps={NQ/med:.0f} "
      f"recall@10_min={min(recalls):.6f} result_md5={res_hash.hexdigest()}", flush=True)
print("THP:", open('/sys/kernel/mm/transparent_hugepage/enabled').read().strip(), flush=True)
