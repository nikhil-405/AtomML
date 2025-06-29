# AtomML

[![CI](https://github.com/nikhil-405/AtomML/actions/workflows/python-app.yml/badge.svg)](https://github.com/nikhil-405/AtomML/actions/workflows/python-app.yml)
[![Coverage Status](https://codecov.io/gh/nikhil-405/AtomML/branch/main/graph/badge.svg)](https://codecov.io/gh/nikhil-405/AtomML)


AtomML is a lightweight, modular machine-learning library built from scratch on NumPy and Pandas. It includes simple and highly unoptimized implementations of various Machine Learning algorithms ```from scratch``` without the use of scikit-learn, TensorFlow, or PyTorch.

## Installation

```bash
# Clone the repo
git clone https://github.com/nikhil-405/AtomML.git
cd AtomML
```

```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate
```

```bash
# Install dependencies
pip install -r requirements.txt
````

```bash
# Set up modules and submodules
pip install -e . 
```

## Tests
If you wish to test the Library, you can run the tests using pytest:

```bash
# Install dependencies
pip install -r requirements.txt
````

```bash
# Set up modules and submodules
pip install -e . 
```


For testing the entire library, you can run:

```bash
pytest tests
```

If you wish to test individual components of the Library, you can run the tests using pytest:

```bash
pytest tests/<test_file>.py
```

## References
The library is inspired by various sources, including but not limited to:
- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)

- [Deep Learning by Ian Goodfellow & co.](https://www.deeplearningbook.org/)

- [Scikit-learn Source Code](https://github.com/scikit-learn/scikit-learn/)

- [Torch Source Code](https://github.com/pytorch/pytorch/)
