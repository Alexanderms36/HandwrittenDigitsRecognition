<h1 align="center">HandwrittenDigitsRecognition</h1>

---

This repository contains an implementation of a neural network based on the material from Michael Nielsen's book Neural Networks and Deep Learning. The project includes essential algorithms for training neural networks, such as backpropagation, gradient descent, and hyperparameter tuning. The convolutional network was remade for aesara (it used to be theano in the original book: http://neuralnetworksanddeeplearning.com/).

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running the network from chapter 1](#running-the-chapter1)
  - [Running the network from chapter 3](#running-the-chapter3)
  - [Running the network from chapter 6](#running-the-chapter6)
    - [Expand data](#expand-data)
    - [Running the network](#running-net)
- [File Structure](#file-structure)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Alexanderms36/HandwrittenDigitsRecognition.git
   cd HandwrittenDigitsRecognition
   ```

2. Create a virtual environment (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # for Windows: venv\\Scripts\\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure the following libraries are included in your `requirements.txt`:
   - `python 3.8.20`
   - `aesara 2.9.4`
   - `numpy 1.19.5`

## Usage

### Running the network from chapter 1

To run the MLP from chapter 1:
```py
py train_network1.py
```

### Running the network from chapter 3



### Running the network from chapter 6

It's useful to use expanded dataset to achieve a better accuracy. You can skip the next subparagraph if it's needed.

#### Expand data

Open `expand_mnist.py` in IDE or code editor.
Make sure the path exists. If it's not, you can change the path to the new dataset. Change the string in the quotes here : 
```py
if os.path.exists("data/mnist_expanded.pkl.gz")
```

Than run the `expand_mnist.py` with the following command:
```bash
py expand_mnist.py
```
Or use methods from your IDE.

#### Running the Script

Now you can use `train_network3.py` to run the code from chapter 6 (Convolutional net)
```bash
py train_network3.py
```

Make sure the path to the data is correct:
```py
training_data, validation_data, test_data = cnn.load_data_shared("./data/mnist_expanded.pkl.gz")
```

## File Structure

<!-- - `function.py`: Main Python script that contains the `draw_plots` function
- `test_function.py`: Unit tests to verify the functionality of the methods in `function.py`
- `Notebook.ipynb`: Jupyter notebook for interactive data analysis and visualization
- `plots/`: Directory where generated plots are saved as .png pictures
- `requirements.txt`: List of dependencies required to run the project
- `deviation.json`: JSON data using for drawing plots -->