# simple speech commands
A pretrained Pytorch classifier for the Google Speech Commands dataset (and the digit subset) that is very quick to set up and use.

## What?
The [Google Speech Commands](https://arxiv.org/abs/1804.03209) dataset is a 16kHz audio dataset of 1s clips of people saying 1 of 35 possible words, which act as classes in a classification task. 
A classification model takes as input a 1s audio clip and identifies which word is spoken (out of the 35 possible words).
While many classification models have already been trained for this well-studied task, few were simple 'plug-and-play' solutions. This repository aims to be that plug-and-play solution. 

## Quick start
No repository cloning or downloading notebooks needed! To perform inference with the pretrained model, simply:
1. Ensure `pytorch` (>=1.10), `torchaudio`, `numpy`, and `omegaconf` are installed (`sklearn`, `matplotlib`, `tensorboard`, `fastprogress`, `pandas` are also needed for training).
2. Run:
    ```python
    import torch
    import torchaudio
    
    model = torch.hub.load('RF5/simple-speech-commands', 'convgru_classifier')
    # or 'convgru_classifier_sc09' if you want the classifier only trained on the 10 digits
    # for 'convgru_classifier_sc09' you can load either the last checkpoint 
    #   model = torch.hub.load('RF5/simple-speech-commands', 'convgru_classifier_sc09', type='last')
    # or the best checkpoint (default)
    #   model = torch.hub.load('RF5/simple-speech-commands', 'convgru_classifier_sc09', type='best')
    model.eval()

    x, _ = torchaudio.load("<path to 16kHz audio file>.wav") # x has shape (1, T)
    x_lens = torch.tensor((x.shape[-1],), dtype=torch.long) # a (1,) tensor of lengths of x

    with torch.no_grad():
        logits = model(x, x_lens)[0] 
        # logits is a (n_classes,) vector of logits
    
    probabilities = logits.softmax(dim=-1)
    # now probabilities[i] is the predicted probability of class[i]
    predicted_class = model.classes[int(probabilities.argmax())] # e.g. "three"
    ```

    The model takes in floating point 16kHz waveforms of the shape `(batch size, samples)` and the length of each waveform in the batch as a long tensor of shape `(batch size,)`. The model returns an output of logits of shape `(batch size, num classes)` over each class. The mapping between class names and indices are given as an ordered list of class names `model.classes` .

Trivial!

## Model
The model is defined in `convrnn_classifier.py` and it consists of an initial convolutional encoder (very similar to that used by wav2vec 2.0 or [HuBERT](https://github.com/pytorch/fairseq/tree/main/examples/hubert)) into a 3-layer, 768-hidden-dimensional GRU. 
The output of the final GRU layer is fed into a final linear layer to yield the logits over the 35 classes (or 10 classes for the digits subset). 

The ConvGRU model only assumes that any input waveform are floating point numbers between -1.0 and 1.0 sampled at 16kHz. You may wish to normalize the volume to ensure very soft audio is scaled up, but it is not strictly necessary and performance will still be good.

Two checkpoints are available under the releases tab -- one model trained on all 35 words (`convgru_classifier`), and another trained on the 10 digits subset -- i.e. it is trained to only classify an utterance as saying one of the ten possible digits (`convgru_classifier_sc09`). 

## Performance

The models are evaluated on the official validation and test sets of the Google Speech Commands dataset. We report accuracies on these below:

| model | dataset | valid set accuracy | test set accuracy |
| ----------- | --- | :-----------: | :----: |
| `convgru_classifier`| Speech Commands full dataset | 92.7% | 92.0% |
| `convgru_classifier_sc09` (best validation loss) | Speech Commands digits (SC09) subset | 96.6% | 96.1% |
| `convgru_classifier_sc09` (last) | Speech Commands digits (SC09) subset | 97.4% | 96.5% |


## Training
If you wish to train your own model, use the `train.py` script. 
It uses `omegaconf` to manage config option and needs the additional dependencies specified earlier.
To see all training configuration options, see `python train.py --help`. All config options can be overridden/specified with command line options. 

To repeat the training with default parameters:
1. Construct training, validation, and testing csv files with `split_data.py`. Simply run `python split_data.py --root_path <path to google speech command dataset directory>` . This will produce csv files of format: `path,label` containing the file path and class label of each utterance in the set. There is also an optional CLI argument to use the SC09 digits subset, if you wish to train a model on the just classifying the 10 digits.
2. Run:
   ```
    python train.py checkpoint_path=runs/run1 train_csv=splits/train.csv
                   valid_csv=splits/valid.csv
   ```
   And it will save the checkpoints and tensorboard logs in the `checkpoint_path` folder. 

## Evaluation
This is done by `eval_test.py`, and it has 3 command line arguments to specify the test csv, checkpoint path, and device. Check out the `python eval_test.py --help` for more details. It simply computes the accuracy on the test set and saves a new checkpoint file with the optimizer state dict stripped out. 

