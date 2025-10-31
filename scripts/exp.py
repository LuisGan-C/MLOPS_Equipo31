import os, yaml
p = "configs/train.yaml"
d = yaml.safe_load(open(p))
N = os.getenv("N")
D = os.getenv("D")
if N: d["model"]["n_estimators"] = int(N)
if D: d["model"]["max_depth"]   = int(D)
open(p, "w").write(yaml.safe_dump(d, sort_keys=False))
print("YAML actualizado:", d["model"])
