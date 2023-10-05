import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class iris_classifier:
    def __init__(self, model = None, train_loader = None, test_loader = None, epochs = None):
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.model  = model
        self.EPOCHS = epochs
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer     = optim.Adam(params = model.parameters(), lr = 0.001)
    
    def start_training(self):
        history    = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        TRAIN_LOSS = []
        VAL_LOSS   = []
        TRAIN_ACCURACY = []
        VAL_ACCURACY   = []

        # train the model
        self.model.train()
        # Run a loop with respect to defined Epoch
        for epoch in range(self.EPOCHS):
            """
                1. Extract the data(X_batch), label(y_batch) from the `train_loader`
                2. Pass X_batch as a training data into the model and do the prediction
                3. Compute the Loss Function
                4. Store computed loss into TRAIN_LOSS
            """
            for (X_batch, y_batch) in self.train_loader:
                y_batch = y_batch.long()
                # Do the prediction
                train_prediction = self.model(X_batch)
                # Compute the loss with the predicted and orginal
                train_loss = self.loss_function(train_prediction, y_batch)
                """
                    1. Initiate the Optimizer
                    2. Do the backward propagation with respect to train_loss
                    3. Do the step with optimizer
                """
                # Initialize the optimizer
                self.optimizer.zero_grad()
                # Do back propagation
                train_loss.backward()
                # Do the step with respect to optimizer
                self.optimizer.step()

            # Do the prediction of training
            train_predicted = torch.argmax(train_prediction, dim = 1)
            # Append the train accuracy
            TRAIN_ACCURACY.append(accuracy_score(train_predicted, y_batch))
            # Append the train loss
            history['accuracy'].append(accuracy_score(train_predicted, y_batch))
            
            with torch.no_grad():
                # Append the train loss
                TRAIN_LOSS.append(train_loss.item())
                # Append the train loss into the history
                history['loss'].append(train_loss.item())

            ########################
            #       Testing        #
            ########################

            """
                1. Extract the data(val_batch), label(val_batch) from the `test_loader`
                2. Pass val_batch as a training data into the model and do the prediction
                3. Compute the Loss Function
                4. Store computed loss into VAL_LOSS & VAL_ACCURACY
            """
            # Run a loop with respect to test_loader
            for (val_data, val_label) in self.test_loader:
                val_label = val_label.long()
                # Do the prediction
                test_prediction = self.model(val_data)
                # Compute the loss
                test_loss = self.loss_function(test_prediction, val_label)

            # Append the test loss
            with torch.no_grad():
                VAL_LOSS.append(test_loss.item())
                history['val_loss'].append(test_loss.item())
                # Compute the accuracy
                test_predicted = torch.argmax(test_prediction, dim = 1)
                # Append the accuracy of testing data
                VAL_ACCURACY.append(accuracy_score(test_predicted, val_label))
                history['val_accuracy'].append(accuracy_score(test_predicted, val_label))

            #########################
            #        Display        #
            #########################

            print("Epoch {}/{} ".format(epoch + 1, self.EPOCHS))
            print("{}/{} [=========================] loss: {} - accuracy: {} - val_loss: {} - val_accuracy: {} ".format(self.train_loader.batch_size,\
                                                                                                                        self.train_loader.batch_size,\
                                                                                                                        np.array(train_loss.item()).mean(),
                                                                                                                        accuracy_score(train_predicted, y_batch),\
                                                                                                                        np.array(test_loss.item()).mean(),\
                                                                                                                        accuracy_score(test_predicted, val_label)))
            
            