# TSAI EVA8
## Phase 1 - Assignment 3 [Part 1]
---
## Link to Excel Sheet
<br/>

[Link](https://docs.google.com/spreadsheets/d/1kS1LgLNPGfsphP72MyauhxoZlSMNs60iIjGnWhl3Jjc/edit?usp=sharing)

---
## Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github.
<br/>

Here we show a sample network for the purpose of this exercise. For the simplicity of calculation we avoid the usage of biases in the network.

![](./assets/samplenetwork.png)

<br/>

Below is the image of the backpropagation (created in [excel](https://docs.google.com/spreadsheets/d/1kS1LgLNPGfsphP72MyauhxoZlSMNs60iIjGnWhl3Jjc/edit?usp=sharing)) that was done in the session.

<br/>

![](./assets/backprop_ss.png)
Here we are going through 100 iterations of backpropagation and plotting th chart below.

<br/>

The learning rate (η) taken here is 2.0 and the activation function for this calculation exercise is fixed at sigmoid (σ) [1/(1+e^(-x)].

<br/>

![](./assets/lr_2.0.png)

<br/>

Here we have used the exact same parameter values that has been discussed in the session. Below listed are the initial values and brief description for each of the acronyms. (based on the network mentioned in the image)
1. i1 = 0.05 (input 1 - 1st input provided to the network).
2. i2 = 0.10 (input 2 - 2nd input provided to the network).
3. t1 = 0.01 (target 1 - 1st Actual Value).
4. t2 = 0.99 (target 2 - 2nd Actual Value).
5. h1 = 1st neuron of the hidden layer.
6. h2 = 2nd neuron of the hidden layer.
7. w1 = 0.15 (w1 is the weight connection between input 1 [i1] to 1st neuron of the hidden layer [h1]).
8. w2 = 0.20 (w2 is the weight connection between input 2 [i2] to 1st neuron of the hidden layer [h1]).
9. w3 = 0.25 (w3 is the weight connection between input 1 [i1] to 2nd neuron of the hidden layer [h2]).
10. w4 = 0.30 (w4 is the weight connection between input 2 [i2] to 2nd neuron of the hidden layer [h2]).
11. a_h1 = 1st neuron of the hidden layer activated by some activation funtion, in this case sigmoid (σ).
12. a_h2 = 2nd neuron of the hidden layer activated by some activation funtion, in this case sigmoid (σ).
13. O1 = 1st neuron of the output layer.
14. O2 = 2nd neuron of the output layer.
15. w5 = 0.40 (w5 is the weight connection between activated 1st hidden neuron [a_h1] to 1st neuron of the output layer [O1]).
16. w6 = 0.45 (w6 is the weight connection between activated 2nd hidden neuron [a_h2] to 1st neuron of the output layer [O1]).
17. w7 = 0.50 (w7 is the weight connection between activated 1st hidden neuron [a_h1] to 2nd neuron of the output layer [O2])
18. w8 = 0.55 (w8 is the weight connection between activated 2nd hidden neuron [a_h2] to 2nd neuron of the output layer [O2])
19. a_O1 = 1st neuron of the Output layer activated by some activation funtion, in this case sigmoid (σ). 1st Predicted value of the network.
20. a_O2 = 2nd neuron of the Output layer activated by some activation funtion, in this case sigmoid (σ). 2nd Predicted value of the network.
21. E1 = Error or loss from the 1st activated Output Neuron, defined as a function of difference between prediction and actual.
22. E2 = Error or loss from the 2nd activated Output Neuron, defined as a function of difference between prediction and actual.
23. E_Total = Total Error or loss of the network, i.e., Total of the error from both the output neurons.
24. η = Learning Rate of the network.

<br/>

In the above description i1, i2, t1, t2 remain constant throughout. Other metrics are parameters which change through the process of back propagation while training the network.
Learning Rate (η) of the network is a hyperparameter along with the activation function which can be set by the user prior to training the network.

<br/>

---
## Formulae for calculation of the above parameters only (forward propagation)
```
  h1 = w1 * i1 + w2 * i2
  h2 = w3 * i1 + w4 * i2
  a_h1 = σ(h1)
  a_h2 = σ(h2)
  o1 = w5 * a_h1 + w6 * a_h2
  o2 = w7 * a_h1 + w8 * a_h2
  a_o1 = σ(o1)
  a_o2 = σ(o2)
  E1 = 0.5 * (t1 - a_o1)^2
  E2 = 0.5 * (t2 - a_o2)^2
  E_Total = E1 + E2
```

<br/>


σ above represents sigmoid activation function [Formula: 1/(1+e^(-x)]

<br/>

---
## Derivation of backpropagation (Explain each major step)
Since our primary objective is to reduce error or the difference between prediction and actual output (remaining constant, denoted by t1 and t2), we need to change our parameters/weights (w1 to w8) as they are affecting the prediction value. This in turn changes the values of h1, h2 and O1, O2 (O1, O2 being the pre-activated predicted values).

Backpropagation calculates the gradients of error function with respect to weights. The calculation proceeds backwards through the network changing the weights so as to reduce the error. Here we assume that when one of the weights is changed the other weights remain constant.

Now lets go ahead and canculate the gradient of error with respect to different parameters of the network.
```
  δE_Total/δw5 = δ(E1 + E2)/δw5 ... (1)
  δ(E1 + E2)/δw5 = δ(E1)/δw5 ... (2)
  δ(E1)/δw5 = (δE1/δa_o1)*(δa_o1/δo1)*(δo1/δw5) ... (3)
```
Here in eq (2) we remove E2 since in the network above E2 not getting generated by w5 and hence the gradient for E2 w.r.t. w5 will be 0. Hence, applying the chain rule to eq (2) we get eq (3).
```
  δE1/δa_o1 = δ(0.5 * (t1 - a_o1)^2) / δa_o1 = (t1 - a_o1)*(-1) = a_o1 - t1 ... (3.1)
  δa_o1/δo1 = δ(σ(o1))/δo1 = σ(o1) * (1-σ(o1)) = a_o1 * (1-a_o1) ... (3.2)
  δo1/δw5 = a_h1 ... (3.3)

  Hence,
  δ(E_Total)/δw5 = (a_o1 - t1) * (a_o1 * (1-a_o1)) * (a_h1)
  Similarly, we get
  δ(E_Total)/δw6 = (a_o1 - t1) * (a_o1 * (1-a_o1)) * (a_h2)
  δ(E_Total)/δw7 = (a_o1 - t1) * (a_o1 * (1-a_o2)) * (a_h1)
  δ(E_Total)/δw8 = (a_o1 - t1) * (a_o1 * (1-a_o2)) * (a_h2)
```
Breaking eq (3) above into each of the components we get the above equations (3.x).

```
  δE_Total/δa_h2 =  (a_o2 - t2) * (a_o2) * (1-a_o2) * w8 + (a_o1 - t1) * (a_o1) * (1-a_o1) * w6

  δE_Total/δa_h1 = δ(E1 + E2)/δa_h1
  δ(E1)/δa_h1 = δ(E1 )/δa_o1 * δa_o1/o1 * δo1/δa_h1 = (a_o1 - t1) * (a_o1) * (1-a_o1) * w5 +  (a_o2 - t2) * (a_o2) * (1-a_o2) * w7

  δE_Total/δw1  = δE_Total/δa_o1 * δa_o1/δo1 * δo1/δa_h1 * δa_h1/δh1 * δh1/δw1
  δE_Total/δw1  = δE_Total/δa_h1  *  δa_h1/δh1  *  δh1/δw1
  δE_Total/δw1  = δE_Total/δa_h1  *  a_h1* (1-a_h1)  *  δh1/δw1
  δE_Total/δw1  = δE_Total/δa_h1  *  a_h1* (1-a_h1)  * i1
  δE_Total/δw2  = δE_Total/δa_h1  *  a_h1* (1-a_h1)  * i2
  δE_Total/δw3  = δE_Total/δa_h2  *  a_h2* (1-a_h2)  * i1
  δE_Total/δw4  = δE_Total/δa_h2  *  a_h2* (1-a_h2)  * i2

  δE_Total/δw1  = ((a_o1 - t1) * (a_o1) * (1-a_o1) * w5 + (a_o2 - t2) * (a_o2) * (1-a_o2) * w7 ) * a_h1 * (1-a_h1) * i1
```
Hence, by the above methods and equations we calculate the gradient of error w.r.t. different weights of the network.

To update the weights we use the formula below
```
  w_new = w_old - (η * δ(E_Total)/δw_old)
```
Here, to update the weight we take current weight and subtract the partial derivative of error function with respect to the current weight. We multiply the derivative of the error function by a selected number to make sure that the new updated weight is minimizing the error function; this number is called learning rate (η).

<br/>

---
## Changes to backward propagation of error with each step when Learning Rate (η) of the network is changed.
### Learning rate changed to 0.1
![](./assets/lr_0.1.png)

<br/>

### Learning rate changed to 0.2
![](./assets/lr_0.2.png)

<br/>

### Learning rate changed to 0.5
![](./assets/lr_0.5.png)

<br/>

### Learning rate changed to 0.8
![](./assets/lr_0.8.png)

<br/>

### Learning rate changed to 1.0
![](./assets/lr_1.0.png)

<br/>

### Learning rate changed to 2.0
![](./assets/lr_2.0.png)

---
