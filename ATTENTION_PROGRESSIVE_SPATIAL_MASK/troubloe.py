# trouble_fix.py
import os, json, pickle
import pandas as pd
from collections import defaultdict

# === EDIT THESE PATHS if different in your environment ===
images_dir = "/workspace/nemo/data/RESEARCH/HMER_MASK_MATCH_HME100K/HME100K/train_images"
ann_dir = "/workspace/nemo/data/RESEARCH/HMER_MASK_MATCH_HME100K/attn_ssl"

# helper
def try_load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def try_load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_targets_from_df(df):
    # try to detect image and caption columns
    cols = [c.lower() for c in df.columns]
    img_col = None
    cap_col = None
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("image","img","file","filename","fname")) and img_col is None:
            img_col = c
        if any(k in lc for k in ("caption","label","text","target","annotation")) and cap_col is None:
            cap_col = c
    if img_col is None or cap_col is None:
        return None
    targets = {}
    for _, row in df.iterrows():
        img = str(row[img_col])
        cap = row[cap_col]
        base = os.path.basename(img)
        noext = os.path.splitext(base)[0]
        targets[base] = cap
        targets[noext] = cap
    return targets

# === Step 1: load images list ===
if not os.path.isdir(images_dir):
    raise SystemExit(f"Images directory not found: {images_dir}")
image_list = sorted(os.listdir(images_dir))
print("Total images found in folder:", len(image_list))

# === Step 2: search for annotation files and try to load ===
ann_files = sorted(os.listdir(ann_dir)) if os.path.isdir(ann_dir) else []
print("Annotation dir:", ann_dir, " ->", len(ann_files), "files found")

candidates = [f for f in ann_files if f.lower().endswith((".json",".pkl",".pickle",".csv",".tsv"))]
print("Candidate annotation files:", candidates)

targets = None
for fname in candidates:
    path = os.path.join(ann_dir, fname)
    print("\nTrying to load:", fname)
    try:
        if fname.lower().endswith(".json"):
            obj = try_load_json(path)
            # If it's a dict, assume mapping
            if isinstance(obj, dict):
                targets = {}
                for k,v in obj.items():
                    base = os.path.basename(k)
                    noext = os.path.splitext(base)[0]
                    targets[base] = v
                    targets[noext] = v
                print("Loaded JSON dict, keys:", len(targets))
                break
            # If it's a list of records, try to detect image/caption keys
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                df = pd.DataFrame(obj)
                t = build_targets_from_df(df)
                if t:
                    targets = t
                    print("Loaded JSON list -> DataFrame -> targets len:", len(targets))
                    break
                else:
                    print("JSON list loaded but couldn't detect image/caption columns.")
        elif fname.lower().endswith((".pkl",".pickle")):
            obj = try_load_pickle(path)
            print("Pickle type:", type(obj))
            if isinstance(obj, dict):
                targets = {}
                for k,v in obj.items():
                    base = os.path.basename(k)
                    noext = os.path.splitext(base)[0]
                    targets[base] = v
                    targets[noext] = v
                print("Loaded pickle dict, keys:", len(targets))
                break
            # if pandas DataFrame
            if isinstance(obj, pd.DataFrame):
                t = build_targets_from_df(obj)
                if t:
                    targets = t
                    print("Loaded pickle DataFrame -> targets len:", len(targets))
                    break
            # other list-of-dicts
            if isinstance(obj, list) and len(obj)>0 and isinstance(obj[0], dict):
                df = pd.DataFrame(obj)
                t = build_targets_from_df(df)
                if t:
                    targets = t
                    print("Loaded pickle list -> DataFrame -> targets len:", len(targets))
                    break
            print("Pickle loaded but format not recognized.")
        elif fname.lower().endswith(".csv") or fname.lower().endswith(".tsv"):
            sep = "," if fname.lower().endswith(".csv") else "\t"
            df = pd.read_csv(path, sep=sep, low_memory=False)
            t = build_targets_from_df(df)
            if t:
                targets = t
                print("Loaded CSV -> targets len:", len(targets))
                break
            else:
                print("CSV loaded but couldn't detect image/caption columns. Columns:", df.columns.tolist())
    except Exception as e:
        print("Failed to load", fname, "->", repr(e))

if targets is None:
    print("\nNo usable annotation file found. Please confirm annotations exist in", ann_dir)
    raise SystemExit("No targets loaded - aborting.")

# === Step 3: analyze mismatches ===
target_keys = set(targets.keys())
print("\nSample target keys (first 20):", list(target_keys)[:20])
print("Total unique target keys (count):", len(target_keys))

missing = []
for img in image_list:
    base = os.path.basename(img)
    noext = os.path.splitext(base)[0]
    variants = {img, base, noext, noext + ".jpg", noext + ".png", noext + ".jpeg"}
    if not any(v in target_keys for v in variants):
        missing.append(img)

print("\nImages with NO matching annotation keys:", len(missing))
print("First 30 missing examples:", missing[:30])
# Also print examples of keys that DO match
matched = [img for img in image_list if img in target_keys or os.path.basename(img) in target_keys or os.path.splitext(os.path.basename(img))[0] in target_keys]
print("Images that matched annotations:", len(matched))

# Suggest normalization that would help
if len(missing) > 0:
    print("\nSuggestion: If your dataset uses basenames WITHOUT extensions in annotations, "
          "either update the annotations to include extensions or update your DataPreperation to store no-extension keys.")
else:
    print("\nAll images have at least one matching annotation key. Good!")
