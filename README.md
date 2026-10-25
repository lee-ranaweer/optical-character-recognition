# Handwriting OCR Web App

Draw a word in your browser and a neural network reads it. The model is trained in TensorFlow/Keras on the IAM Handwriting Database and runs entirely client-side with TensorFlow.js, so there is no server and no data leaves your device.

**Live demo:** [add link]

## Results

| Metric | Value |
| --- | --- |
| Character error rate (held-out test set) | **5.4%** |
| Training data | 96,454 labeled word images (IAM) |

The split into training, validation, and test sets was random, so writers can appear in more than one set. Accuracy on a completely new writer, or on mouse-drawn input, is likely lower.

## How it works

1. **Preprocessing.** The drawing is cropped tightly around the ink, converted to grayscale, inverted (ink = 1, paper = 0), and resized with padding to 64 x 256 pixels.
2. **CNN.** Three convolution blocks (32, 64, 128 filters) learn pen-stroke patterns.
3. **Reshape.** The feature map is read left to right as 64 time steps, one slice of the word each.
4. **BiLSTM.** Two bidirectional LSTM layers model letter order using context from both directions.
5. **Output.** Each time step scores every character (79) plus a "blank" symbol.
6. **CTC decoding.** The best character per step is taken, repeats are merged, and blanks are removed to produce the text.

The model is trained with **CTC loss**, which lets it learn from the image and the word alone, without anyone marking where each letter sits.

## Running in the browser

The standard TensorFlow.js converter did not work with the training environment, so the trained weights are exported as raw files and the same network is rebuilt in TensorFlow.js, then loaded layer by layer. On load, the page runs a blank image through the network and compares the result with the Python model's output (`check.json`) to confirm they match.

## Run locally

```bash
python -m http.server
```

Then open http://localhost:8000. The page needs these files next to it:

```
index.html
chars.json      character list used in training
weights.bin     trained weights (raw float32)
weights.json    shape of each weight array
check.json      reference output for the load check
```

Opening `index.html` directly from disk will not work, because browsers block loading the weight files that way.

## Training

Training was done on Google Colab with a GPU.

1. Register for and download the IAM words dataset (`words.tgz`, `ascii.tgz`) from the [IAM Handwriting Database](https://fki.tic.unibe.ch/databases/iam-handwriting-database). The dataset is not included in this repository.
2. Run `handwriting_crnn_ctc.py`: Adam optimizer, early stopping, and learning-rate reduction on plateau.
3. Run `export_weights.py` to write `weights.bin`, `weights.json`, `check.json`, and `chars.json`.

## Limitations

- Reads one handwritten word at a time, not lines or full pages.
- Trained on pen-on-paper scans, so mouse-drawn words are harder for it. Some letters (for example "k") are misread more often.
- Supports only the characters in the IAM dataset.

## Possible improvements

- Data augmentation, including varied stroke thickness
- Fine-tuning on drawings collected from the web app
- Splitting train and test data by writer for a stricter evaluation
- Dictionary-based correction of predictions
- Line-level recognition

## Credits

Trained on the [IAM Handwriting Database](https://fki.tic.unibe.ch/databases/iam-handwriting-database) (U. Marti and H. Bunke). Check the dataset's terms before redistributing any of its images.