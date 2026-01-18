__This CNN vs MLP is small get-started project in Deep Learning__



__High-level flow in main.py:__

1. Read configs/default.yaml for hyperparameters such as batch_size, epochs, lr, and num_workers.
2. Select device as CUDA if available, otherwise CPU (torch.device("cuda" if torch.cuda.is_available() else "cpu")).
3. Call get_mnist_loaders from data.mnist to obtain train_loader and test_loader with configured batch size and worker count.
4. Choose model: if cfg["model"]["name"] == "mlp" instantiate MLP(), else instantiate CNN(); push model to the selected device.
5. Create Adam optimizer over all model parameters with learning rate from config, and CrossEntropyLoss as the criterion.
6. Wrap these in a Trainer object from trainers.trainer which encapsulates training and evaluation logic.
7. For each epoch, call trainer.train_epoch(train_loader) to obtain average training loss and trainer.evaluate(test_loader) to obtain test accuracy, then print a formatted log line.


### MLP model design

 The MLP is a straightforward fully connected network mapping flattened MNIST images (28×28) to 10 class logits. It has two hidden layers with ReLU activations and no explicit regularization.

Architecture in `models/mlp.py`:  
- Constructor signature: `MLP(input_dim=28*28, hidden_dim=256, num_classes=10)`.  
- Layers:  
  - `fc1`: `nn.Linear(input_dim, hidden_dim)`  
  - `fc2`: `nn.Linear(hidden_dim, hidden_dim)`  
  - `fc3`: `nn.Linear(hidden_dim, num_classes)`  
- Forward pass:  
  - Flatten input to shape `(batch_size, 784)` via `x.view(x.size(0), -1)`.  
  - Apply ReLU after `fc1` and `fc2`: `x = F.relu(self.fc1(x))`, `x = F.relu(self.fc2(x))`.  
  - Output raw logits from `fc3` without softmax (paired with `CrossEntropyLoss`).

Conceptually, this model treats each pixel independently, ignoring 2D spatial structure, which is expected to limit performance on image data compared to a CNN.

### CNN model design

The CNN uses convolutional layers with pooling to exploit spatial structure in MNIST images, followed by fully connected layers for classification. It is shallow but sufficient to show advantages over an MLP on this dataset.

Architecture in `models/cnn.py`:  
- Constructor signature: `CNN(num_classes=10)`.  
- Layers:  
  - `conv1`: `nn.Conv2d(1, 32, kernel_size=3, padding=1)`; keeps spatial size at 28×28.  
  - `conv2`: `nn.Conv2d(32, 64, kernel_size=3, padding=1)`.  
  - `pool`: `nn.MaxPool2d(2, 2)` for downsampling.  
  - `fc1`: `nn.Linear(64 * 7 * 7, 128)`; the input size corresponds to two successive 2×2 poolings from an initial 28×28 image.  
  - `fc2`: `nn.Linear(128, num_classes)`.  
- Forward pass:  
  - Apply `conv1` → ReLU → max pool: input `(N, 1, 28, 28)` becomes `(N, 32, 14, 14)`.  
  - Apply `conv2` → ReLU → max pool: becomes `(N, 64, 7, 7)`.  
  - Flatten: `x.view(x.size(0), -1)` to `(N, 64 * 7 * 7)`.  
  - Apply `fc1` + ReLU, then `fc2` to produce logits.

This network encodes locality via convolutions, reduces dimensionality via pooling, and then classifies based on learned spatial features.

#### Training loop and configuration behavior

The training logic uses a `Trainer` class which is invoked with consistent methods: `.train_epoch()` and `.evaluate()`. This standardizes training across different architectures and allows the same loop to be reused for both MLP and CNN models.

Key details in `main.py` that determine training behavior:  
- Optimizer: `optim.Adam(model.parameters(), lr=cfg["training"]["lr"])`.  
- Loss: `nn.CrossEntropyLoss()`, appropriate for multi-class classification with logits.  
- Epoch logging:  
  - Prints epoch index, total epochs from config, training loss, and test accuracy in percent.


| Aspect               | MLP implementation                               | CNN implementation                                                |
|----------------------|--------------------------------------------------|-------------------------------------------------------------------|
| Input handling       | Flattens image to 784-dimensional vector. | Keeps 2D structure as 1×28×28 tensor.                    |
| Inductive bias       | No spatial locality, treats all pixels equally. | Encodes locality via convolution and pooling.           |
| Feature extraction   | Fully connected layers learn global features. | Convolutions learn local patterns, pooling builds hierarchies. |
| Parameters   |  |  |
| Expected performance | Adequate but limited on images.          | Typically higher accuracy on MNIST with same training setup. |
-----