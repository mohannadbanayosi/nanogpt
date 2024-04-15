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

Example generated text:
```
let no mother. No, Lord Richard, if I say I
Wish of your offended disposition: we unto pardon
For Edward for an ill:the fair Romans' him:
Romeo, words be sometime known apparent,
Adbandine widow the peer in his dayers, away,
My faults are doing commission with his weddess'd;
Bearfulnce shorted Boyingbroke, that he before these
The instructed sin, an if despise he were ay,
Tender the Edward thought of Earl of Menenius,
And wast of that she handship the husband all.
Give me his father? Thy valour, that found to last a
Thow swords be compedion'd wind feeling far in death.

MARIANA:
Ay, how! fall would the caln against an heaven!

PAULINA:
Here a mistress'd is in the noble power by hence;
And that this enemy's chair in the choices of my day,
Reverence and war a help Keeperland pieces
To change, in bless and whrt it feel short
She sad ho stand of mine est life.
O, then, to make him!
Fash a brother's a life, till you much lamb wished!

Second Set:
What, gentlement once here new to mine honour'd,
Proceedine than the fill'd for me and majemy
Of York's uncle: yield me for a few-credit,
Ssent disguadined wild taked me another of England's
Despised thy nearer dispositence of the heum?

KING RICHARD III:
Have imposed me to it lamentable bloody!

QUEEN ELIZABETH:
And he did, yet, hear I'll rare you, my wrathest.

QUEEN ELIZABETH:
He hath there.

GLOUCESTER:
My lord!
```
