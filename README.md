# nanogpt

Building a nanogpt following @karpathy's lectures. Check [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).

## Setup

```bash
pip install -r requirements.txt
```

If you want to use [mps on macbook](https://developer.apple.com/metal/pytorch/), select `Preview (Nightly)` when installing Pytorch from [this resource](https://pytorch.org/get-started/locally/).

Command will look like this:

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Train

The model is configured to train on text under [input.txt](input.txt), which is a copy of [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). Feel free to replace the training dataset with anything else.

```bash
python train.py
```

After the model is trained a `.pth` file will be saved under [model_resources/](model_resources/).

## Generate Text

```bash
python generate.py
```
