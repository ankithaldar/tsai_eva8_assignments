# TSAI EVA8
## Phase 1 - Assignment 2.5
---
## Architecture used
```mermaid
  flowchart TB
    subgraph one [INPUTS]
      direction TB
      input1[MNIST Input Image]
      input2[Random Number]
    end
    subgraph two [LENET]
      direction TB
      input1 --> CONV1_Relu
      CONV1_Relu --> CONV2_Relu
      CONV2_Relu --> AvgPool1
      AvgPool1 --> CONV3_Relu
      CONV3_Relu --> CONV4_Relu
      CONV4_Relu --> AvgPool2
      AvgPool2 --> Flatten
      Flatten --> FC1_Sigmoid
      FC1_Sigmoid --> FC2_Sigmoid
      FC2_Sigmoid --> FC3_Softmax
    end

    subgraph three [SumPredNET]
      direction LR
      FC3_Softmax --> torch.stack
      input2 --> torch.stack
      torch.stack --> FC_A_1_Sigmoid --> FC_A_2_Sigmoid --> FC_A_3_Sigmoid
    end

    subgraph four [FINAL OUTPUTS]
      direction LR
      FC3_Softmax --> label[MNIST Predicted Label]
      FC_A_3_Sigmoid --> sum[SUM of MNIST Predicted Label + RandomNumber]
    end
```
---
## Data representation
<br/>
MNIST data is downloaded from torchvision dataset link in the format of training.pt and test.pt where both files are numpy Binary file. Here, the images have 1 channel represented by a 28x28 matrix. The number of images are 6000 per label. There are a total of 10 labels (0-9), hence 60000 total images in the train dataset. Similarly, we have a separate test dataset of 10000 images with equal bin size per label.

Random numbers are generated for each individual labels of train and test set. This is fed into the model via DataLoaders (torch.data.utils).

---
## How have we combined the two inputs?
<br/>
We pass the image and the randomly generated number to the model as inputs, where we predict the input to the label and then stack the MNIST predicted label with the randomly generated number assigned for that image. This has been done inside the model to create a dataset for further prediction of the summation function.

---
## How are we evaluating our results?
<br/>
To evaluate the results, we calculate the total number of predictions that exactly match the labels (torchmetrics.Accuracy) <i>MNIST predicted label</i> and the <i>MNIST + random number predicted label</i>. Further, we add the loss from both prdeictions to understand how the loss is being minimized.

---
## What results did we finally get and how did we evaluate our results?
<br/>

```
Testing LeNet...
Epoch: 01 | m_mnist_accuracy: 99.14% | m_sum_accuracy: 90.07%
Testing Ended...
```
m_mnist_accuracy are the number of MNIST labels correctly predicted by our model which is 99.13% while m_sum_accuracy is the number of correctly predicted labels for MNIST label + Random number. Our model is correctly predicting it upto 87.05%

---
## Training Logs
### Image Logs
<br/>
<b>Loss Function for training</b>
<img title="Loss Function for training" alt="Loss Function for training" src="assets/Loss.png">
<br/>
<br/>
<br/>
<b>Accuracy for training</b>
<img title="Accuracy for training" alt="Accuracy for training" src="assets/Metrics.png">

### Text Logs

```
Training on GPU
Training LeNet...
Epoch: 01 | m_mnist_accuracy: 66.46% | m_sum_accuracy: 9.02%
Epoch: 02 | m_mnist_accuracy: 84.59% | m_sum_accuracy: 9.82%
Epoch: 03 | m_mnist_accuracy: 96.67% | m_sum_accuracy: 17.57%
Epoch: 04 | m_mnist_accuracy: 97.53% | m_sum_accuracy: 27.79%
Epoch: 05 | m_mnist_accuracy: 97.99% | m_sum_accuracy: 35.66%
Epoch: 06 | m_mnist_accuracy: 98.24% | m_sum_accuracy: 44.86%
Epoch: 07 | m_mnist_accuracy: 98.47% | m_sum_accuracy: 51.68%
Epoch: 08 | m_mnist_accuracy: 98.69% | m_sum_accuracy: 60.92%
Epoch: 09 | m_mnist_accuracy: 98.76% | m_sum_accuracy: 69.06%
Epoch: 10 | m_mnist_accuracy: 98.84% | m_sum_accuracy: 74.10%
Epoch: 11 | m_mnist_accuracy: 99.02% | m_sum_accuracy: 75.79%
Epoch: 12 | m_mnist_accuracy: 99.08% | m_sum_accuracy: 76.57%
Epoch: 13 | m_mnist_accuracy: 99.17% | m_sum_accuracy: 76.88%
Epoch: 14 | m_mnist_accuracy: 99.20% | m_sum_accuracy: 77.64%
Epoch: 15 | m_mnist_accuracy: 99.24% | m_sum_accuracy: 78.21%
Epoch: 16 | m_mnist_accuracy: 99.29% | m_sum_accuracy: 78.29%
Epoch: 17 | m_mnist_accuracy: 99.36% | m_sum_accuracy: 78.61%
Epoch: 18 | m_mnist_accuracy: 99.42% | m_sum_accuracy: 79.52%
Epoch: 19 | m_mnist_accuracy: 99.43% | m_sum_accuracy: 80.78%
Epoch: 20 | m_mnist_accuracy: 99.44% | m_sum_accuracy: 79.25%
Epoch: 21 | m_mnist_accuracy: 99.50% | m_sum_accuracy: 80.32%
Epoch: 22 | m_mnist_accuracy: 99.48% | m_sum_accuracy: 80.42%
Epoch: 23 | m_mnist_accuracy: 99.52% | m_sum_accuracy: 80.63%
Epoch: 24 | m_mnist_accuracy: 99.54% | m_sum_accuracy: 79.95%
Epoch: 25 | m_mnist_accuracy: 99.54% | m_sum_accuracy: 81.07%
Epoch: 26 | m_mnist_accuracy: 99.59% | m_sum_accuracy: 80.08%
Epoch: 27 | m_mnist_accuracy: 99.60% | m_sum_accuracy: 80.97%
Epoch: 28 | m_mnist_accuracy: 99.62% | m_sum_accuracy: 81.92%
Epoch: 29 | m_mnist_accuracy: 99.62% | m_sum_accuracy: 81.10%
Epoch: 30 | m_mnist_accuracy: 99.65% | m_sum_accuracy: 82.66%
Epoch: 31 | m_mnist_accuracy: 99.64% | m_sum_accuracy: 83.20%
Epoch: 32 | m_mnist_accuracy: 99.68% | m_sum_accuracy: 82.48%
Epoch: 33 | m_mnist_accuracy: 99.69% | m_sum_accuracy: 83.42%
Epoch: 34 | m_mnist_accuracy: 99.69% | m_sum_accuracy: 83.84%
Epoch: 35 | m_mnist_accuracy: 99.67% | m_sum_accuracy: 85.89%
Epoch: 36 | m_mnist_accuracy: 99.70% | m_sum_accuracy: 84.99%
Epoch: 37 | m_mnist_accuracy: 99.73% | m_sum_accuracy: 87.39%
Epoch: 38 | m_mnist_accuracy: 99.72% | m_sum_accuracy: 87.08%
Epoch: 39 | m_mnist_accuracy: 99.71% | m_sum_accuracy: 87.17%
Epoch: 40 | m_mnist_accuracy: 99.74% | m_sum_accuracy: 87.40%
Epoch: 41 | m_mnist_accuracy: 99.71% | m_sum_accuracy: 87.36%
Epoch: 42 | m_mnist_accuracy: 99.76% | m_sum_accuracy: 87.41%
Epoch: 43 | m_mnist_accuracy: 99.73% | m_sum_accuracy: 87.39%
Epoch: 44 | m_mnist_accuracy: 99.73% | m_sum_accuracy: 87.39%
Epoch: 45 | m_mnist_accuracy: 99.70% | m_sum_accuracy: 87.65%
Epoch: 46 | m_mnist_accuracy: 99.72% | m_sum_accuracy: 88.81%
Epoch: 47 | m_mnist_accuracy: 99.74% | m_sum_accuracy: 87.70%
Epoch: 48 | m_mnist_accuracy: 99.78% | m_sum_accuracy: 90.28%
Epoch: 49 | m_mnist_accuracy: 99.75% | m_sum_accuracy: 90.36%
Epoch: 50 | m_mnist_accuracy: 99.72% | m_sum_accuracy: 90.88%
Training Ended...
```

---
## What is the loss function that we picked and why?
<br/>
In the case of multiple-class classification, we can predict a probability for the example belonging to each of the classes, we have used a cross-entropy loss function, also referred to as Logarithmic loss. The problem is framed as predicting the likelihood of an example belonging to each class. We have used cross entropy loss since, the penalty is logarithmic, offering a small score for small differences (0.1 or 0.2) and enormous score for a large difference (0.9 or 1.0). Since the activation function for the output layer is Softmax and it is a continuously differentiable function, this makes it possible to calculate the derivative of the loss function with respect to every weight in the neural network. This property allows the model to adjust the weights accordingly to minimize the loss function (model output close to the true values).

---
## How did we train our Pytorch model on GPU?
<br/>
Attaching a screenshot of model training in GPU.
<br/>
<br/>
<b>Model Training on GPU</b>
<br/>
<img title="Model Training on GPU" alt="Model Training on GPU" src="assets/GPU.png">
<br/>
<br/>
We have used the following code to check that.

```
# Move model to device
self.model = self.model.to(self.device)
print('Training on GPU' if self.model.cuda() else 'Training on CPU')
```
