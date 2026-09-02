# Image Classification Using a Pretrained Model

A tiny, beginner-friendly example that uses **MobileNetV2** (a small pretrained
neural network from PyTorch's `torchvision`) to guess what is in a photo.
It comes as a command-line script, a web app you can upload photos to, and a
JSON API.

## Run it (command line)

```bash
pip install -r requirements.txt
python classify.py                 # classifies every image in ./images
python classify.py path/to/cat.jpg # or pass your own images
```

The first run downloads the model weights (~14 MB). Example output:

```
dog.jpg
  1. Rottweiler                 73.3%
  2. black-and-tan coonhound     9.2%
  3. Labrador retriever          5.5%
```

## Run it (web app)

```bash
python app.py
```

Open <http://localhost:5000>, choose a photo and press **Classify**. The page
shows your image next to the top-5 guesses with confidence bars.

![web app screenshot](docs/webapp.png)

### JSON API

The same server exposes `POST /predict` so other programs (a phone app, a
script, another website) can use the model:

```bash
curl -F "image=@images/dog.jpg" http://localhost:5000/predict
```

```json
{"predictions": [
  {"label": "Rottweiler", "probability": 0.7329},
  {"label": "black-and-tan coonhound", "probability": 0.0918},
  {"label": "Labrador retriever", "probability": 0.0552},
  {"label": "Tibetan mastiff", "probability": 0.0243},
  {"label": "Doberman", "probability": 0.0213}
]}
```

Bad input returns a `400` with an `error` message (wrong file type, corrupt
image, or a file over 10 MB).

## Sample images

The `images/` folder holds a handful of small public-domain photos
(a dog, a cat, a car, a banana, a pizza) plus a few deliberate tricky cases,
`portrait.jpg`, `dogs.jpg` and `fruit.jpg` (see Limitations below). Drop in any `.jpg`/`.png` to try more.
The full output of a run is in `sample_output.txt`.

## How the pretrained model works (in plain English)

1. **It was trained already.** Someone trained MobileNetV2 on ImageNet, a
   dataset of ~1.2 million photos labelled with 1,000 categories ("golden
   retriever", "sports car", "banana", ...). Training took a lot of compute;
   we just download the finished result (the *weights*) and reuse it. That is
   what "pretrained" means.

2. **Preprocessing.** The model only understands one input shape, so every
   image is resized and cropped to 224x224 pixels, converted to numbers
   (0-1 per colour channel), and normalised with the same mean/std used during
   training. `weights.transforms()` does all of this for us.

3. **Layers of filters.** The image passes through many *convolutional*
   layers. Early layers detect simple things (edges, colour blobs); later
   layers combine those into textures, parts (ears, wheels), and eventually
   whole objects. MobileNetV2 is "small" because it uses cheap
   *depthwise-separable* convolutions, so it runs fine on a laptop or phone.

4. **Scores -> probabilities.** The final layer outputs 1,000 raw scores
   (*logits*), one per category. `softmax` squashes them into probabilities
   that add up to 100%, and we print the top 3.

5. **Limitations.** The model can only name things in its 1,000 ImageNet
   classes, so a photo of something unusual will get a "closest match" guess.
   `portrait.jpg` shows this: ImageNet has no "person" class, so a man in
   sunglasses under blue light comes out as `mask 24.9%` / `ski mask 22.0%`.
   Notice the low confidence, which is a useful hint that the model is unsure.
   The model also assumes **one object per image**: `dogs.jpg` (a collage of
   nine breeds) gets `Welsh springer spaniel 55.0%` for just one of them, and
   `fruit.jpg` (a mixed fruit pile) spreads its guesses across
   `custard apple 33.1%`, `acorn squash 16.6%`, `spaghetti squash 11.2%`.
   Detecting several objects at once needs a different kind of model
   (an *object detector*), not a classifier.
   It also has no idea *why* it is right; it has simply learned patterns of
   pixels that correlate with each label.
