# Iris Classification with Neural Network

This repository provides a Python script to train a neural network model for classifying Iris flowers. The dataset used for this classification task is the Iris dataset. This README.md file will guide you through the installation and usage of this script.

## Installation

Before using the script, you need to install the required dependencies. You can use the following commands to install them:

```bash
!pip install -t . git+https://github.com/atikul-islam-sajib/irisClass.git --upgrade
!pip install classifier

or

!pip install irisClass
```

## Usage

Follow these steps to use the script for Iris classification:

1. Import the necessary modules:

```python
import classifier
from classifier.dataset import dataloader
from classifier.ANN import ANN
from classifier.iris_training import iris_classifier
import matplotlib.pyplot as plt
from classifier.evaluation import evaluation
from classifier.KFold import KFold_CV
```

2. Load the Iris dataset:

```python
dataloader = dataloader(dataset='/content/Iris.csv')
X, y, train_loader, test = dataloader.load_data()
```

3. Create an instance of the neural network model:

```python
model = ANN()
```

4. Display the total number of trainable parameters in the model:

```python
model.total_trainable_parameters(model=model)
```

5. Initialize the Iris classifier:

```python
classifier = iris_classifier(model=model, train_loader=train_loader, test_loader=test, epochs=500)
```

6. Start training the model and retrieve the training history:

```python
history = classifier.start_training()
```

7. Plot the training and validation loss:

```python
plt.plot(history['loss'], label='train_loss')
plt.plot(history['val_loss'], label='test_loss')
plt.legend()
plt.show()
```

8. Plot the training and validation accuracy:

```python
plt.plot(history['accuracy'], label='train_accuracy')
plt.plot(history['val_accuracy'], label='test_accuracy')
plt.legend()
plt.show()
```

9. Perform evaluation on the trained model:

```python
evaluation = evaluation(model=model, TRAIN_LOADER=train_loader, TEST_LOADER=test)
evaluation.train_evaluation()
evaluation.validation_evaluation()
```

10. Perform K-Fold Cross Validation:

```python
KFold_CV(model=model, X=X, y=y, epochs=100, fold=5)
```

Now you can use this script to train and evaluate a neural network model for Iris classification. Adjust the parameters and settings as needed for your specific use case.

Please make sure to check the documentation and code comments for more details on each module and function.
