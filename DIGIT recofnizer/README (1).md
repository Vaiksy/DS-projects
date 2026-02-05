# Handwritten Digit Recognition Neural Network (From Scratch)

This project implements a handwritten digit recognition system using a neural network built completely from scratch with Python and NumPy. The main purpose of this project is to understand how neural networks work internally by implementing all the core components manually, without using high-level deep learning frameworks such as TensorFlow or PyTorch.

---

## 🚀 Key Features

- Neural network implemented entirely from scratch  
- Uses only Python and NumPy  
- Fully connected feedforward architecture  
- Manual implementation of forward propagation  
- Manual implementation of backpropagation  
- Gradient descent–based optimization  
- Predicts handwritten digits from 0 to 9  

---

## 🧠 Model Architecture

The neural network consists of three main layers:

- **Input Layer:**  
  784 neurons representing a flattened 28×28 grayscale image  

- **Hidden Layer:**  
  10 neurons using the ReLU activation function  

- **Output Layer:**  
  10 neurons (digits 0–9) using the Softmax activation function  

---

## ⚙️ How the Model Works

### Data Preparation
Each digit image is flattened into a 784-dimensional vector. The pixel values are normalized, and labels are converted into one-hot encoded vectors so they can be used during training.

### Forward Propagation
The input data moves through the network layer by layer. First, it is multiplied by weights, biases are added, and the ReLU activation function is applied in the hidden layer. The result is then passed to the output layer, where Softmax converts the values into probabilities.

```
Z1 = W1 · X + B1
A1 = ReLU(Z1)

Z2 = W2 · A1 + B2
A2 = Softmax(Z2)
```

The output represents the probability of each digit from 0 to 9, and the digit with the highest probability is selected as the prediction.

### Loss Function
The model uses cross-entropy loss to measure how far the predicted probabilities are from the true labels. A lower loss indicates better model performance.

### Backpropagation
Backpropagation is implemented manually by computing gradients of the loss function with respect to weights and biases. These gradients show how each parameter contributes to the error.

### Parameter Updates
Weights and biases are updated using gradient descent:

```
W = W − α · dW
B = B − α · dB
```

where α is the learning rate.

### Training Process
The model is trained over multiple epochs by repeating forward propagation, loss calculation, backpropagation, and parameter updates until the network learns optimal values.

---

## 📈 Predictions

After training, the neural network can take unseen handwritten digit images as input and accurately predict the digit along with confidence scores for each class.

---

## 🛠 Technologies Used

- Python  
- NumPy  
- Linear Algebra  
- Calculus  

No external machine learning or deep learning frameworks were used.

---

## 📂 Project Structure

```
├── digit recognizer.ipynb
├── README.md
```

---

## 🎯 Project Significance

This project demonstrates a strong understanding of neural network fundamentals, mathematical reasoning behind machine learning, and the ability to build models without relying on abstraction-heavy libraries. It is well suited for learning purposes, technical interviews, and portfolio presentation.

---

## 📌 Future Improvements

- Add more hidden layers  
- Improve training accuracy  
- Implement batch training  
- Add visualization for predictions  
