# Run these as Colab cells. No tensorflowjs, no converter, no installs.
# Use a CPU runtime; a GPU isn't needed.
#
# Needed in /content first:  htr_infer_model.keras  and  chars.json

# %% CELL 1 -- export the trained numbers as plain files
import json
import numpy as np
import tensorflow as tf

# Skip this line if `infer_model` already exists in this session.
infer_model = tf.keras.models.load_model("/content/htr_infer_model.keras", compile=False)

arrays = []
for layer in infer_model.layers:
    for w in layer.weights:
        arr = np.asarray(w.numpy() if hasattr(w, "numpy") else w)
        if "seed_generator" in getattr(w, "path", "") or not np.issubdtype(arr.dtype, np.floating):
            continue                                   # dropout bookkeeping, not learned numbers
        arrays.append(arr.astype("<f4"))
        print(f"{layer.name:28} {tuple(arr.shape)}")

with open("/content/weights.bin", "wb") as f:
    for a in arrays:
        f.write(a.tobytes())
json.dump({"shapes": [list(a.shape) for a in arrays]}, open("/content/weights.json", "w"))

# Reference output for an all-zero image; the web page compares against it.
zeros = np.zeros((1, 64, 256, 1), dtype="float32")
first_row = infer_model(zeros, training=False).numpy()[0, 0].tolist()
json.dump({"logits0": first_row}, open("/content/check.json", "w"))

chars = json.load(open("/content/chars.json"))
print(len(arrays), "arrays (expect 31).  Output classes:", arrays[-1].shape[0],
      "| chars + blank:", len(chars) + 1)

# %% CELL 2 -- download
!cd /content && zip -q web_files.zip weights.bin weights.json check.json chars.json
from google.colab import files
files.download("/content/web_files.zip")
