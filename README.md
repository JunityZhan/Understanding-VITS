# Understanding-VITS
In this repository, you'll explore the inner workings of VITS (Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech) through Jupyter Notebooks. You'll dive into topics such as data normalization, the training process, the inference process, and detailed aspects of the model.

## Usage:
Begin by reading through the vits.ipynb file and executing the code line by line. If you're having trouble understanding the purpose of each block, feel free to use debugging tools as necessary. The other files in the repository serve as supporting resources, which will help you closely examine the methods used within the vits.ipynb file.

## Note:

1. In comments, I may add "**Keywords: xxx**". When you see this, it means if you don't have background knowledge that make you hard to understand, just search the keyword to see if you can understand by other tutorials. Or why not ASK **GPT4**?
2. Because of the nature of class, I can not implement a class step by step, by which we can see how the data is flowed, so you may use **DEBUG TOOL**, add watch to the shape of tensor or some data of tensor, so that it may help you to understand.

### vits.ipynb
In vits.ipynb, I will guide you through a complete process of vits, from making dataset and building models, to training and inference. If you follow the steps, you can train a usable vits model and easy to synthesize voice yourself.

### Dataset.ipynb

VITS is an end-to-end Text-to-Speech (TTS) system, and understanding how it transforms raw resources into training data is crucial. In an end-to-end TTS, we use both waveforms and text as input. We'll delve into the process of converting these waveforms and text into suitable datasets.

### Cleaner.ipynb

Cleaner methods are employed to preprocess the text. By converting the text into phonetic symbols, the system can more effectively distinguish between letters with different pronunciations.

### Models.ipynb

VITS composes various models, such as Flow and Self Attention, to see how they are used in VITS, see Models.ipynb.

