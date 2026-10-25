# Optical Character Recognition Machine Learning

A machine learning model that extracts text from handwritten images. The model is trained in TensorFlow/Keras on the IAM Handwriting Database of 96,454 labeled word images. The demo runs entirely client-side with TensorFlow.js.

## How it works

1. **Preprocessing.** Each image is cropped tightly around the ink, converted to grayscale, inverted (ink = 1, paper = 0), and resized with padding to 64 x 256 pixels.
2. **CNN.** Three convolution blocks (32, 64, 128 filters) learn pen-stroke patterns.
3. **Reshape.** The feature map is read left to right as 64 time steps, one slice of the word each.
4. **BiLSTM.** Two bidirectional LSTM layers model letter order using context from both directions.
5. **Output.** Each time step scores every character (79) plus a "blank" symbol.
6. **CTC decoding.** The best character per step is taken, repeats are merged, and blanks are removed to produce the text.

## Running in the browser

The standard trained weights from the TensorFlow model are exported as raw files to be rebuilt in TensorFlow.js, then loaded layer by layer to run locally in the browser.

## Run locally on http://localhost:8000/ in /app

```bash
python -m http.server
```

## Credits

Trained on the [IAM Handwriting Database](https://fki.tic.unibe.ch/databases/iam-handwriting-database) (U. Marti and H. Bunke)