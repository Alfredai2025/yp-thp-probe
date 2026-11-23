#!/usr/bin/env python3
"""THP effect on graph-ANN (hnswlib on SIFT, single thread).
Fingerprint = md5 of all result sets (must be bit-identical across THP arms).
Speed proves nothing without identical fingerprints.
"""
import hnswlib, numpy as np, hashlib, time, os, sys, h5py

URL = "http://anns-benchmarks.com/sift-128-euclidean.hdf5"
PATH = "sift.hdf5"
if not os.path.exists(PATH):
    import urllib.request
    urllib.request.urlretrieve(URL, PATH)
h = h5py.File(PATH, "r")
XB = np.asarray(h['train'][:200000]).astype(np.float32)      # 200k base
XQ = np.asarray(h['test'][:1000]).astype(np.float32)         # 1k queries
GT = np.asarray(h['neighbors'][:1000, :10])                  # top-10 GT

M, EF_C, EF_S = 16, 200, 128
t0 = time.time()
index = hnswlib.Index(space='l2', dim=128)
index.init_index(max_elements=len(XB), M=M, ef_construction=EF_C, random_seed=42)
index.add_items(XB, np.arange(len(XB)), num_threads=1)
index.set_ef(EF_S)
print(f"build {time.time()-t0:.0f}s", flush=True)

# warmup
index.knn_query(XQ[:100], k=10, num_threads=1)
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
    recalls.append(float(np.mean([len(set(labels[i]) & set(GT[i])) / 10 for i in range(len(XQ))])))
med = sorted(lat)[len(lat)//2]
print(f"RESULT reps={reps} med_query_s={med:.4f} qps={len(XQ)/med:.0f} "
      f"recall@10_min={min(recalls):.6f} result_md5={res_hash.hexdigest()}", flush=True)
print("THP:", open('/sys/kernel/mm/transparent_hugepage/enabled').read().strip(), flush=True)
