# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Basic Tensor Operations
# Create a tensor
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
print(f"Tensor x:\n{x}")

# Create a tensor filled with zeros
zeros = torch.zeros((2, 3))
print(f"Zeros tensor:\n{zeros}")

# Create a random tensor
random_tensor = torch.rand((2, 2))
print(f"Random tensor:\n{random_tensor}")

# Element-wise addition
y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float)
result_add = x + y
print(f"Element-wise addition result:\n{result_add}")

# Matrix multiplication
result_mul = torch.matmul(x, y)
print(f"Matrix multiplication result:\n{result_mul}")

# Move tensor to GPU (if available)
x_gpu = x.to(device)
print(f"Tensor x on {device}:\n{x_gpu}")

# 2. Define a Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 64)   # Fully connected layer 2
        self.fc3 = nn.Linear(64, 10)    # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model and move it to the appropriate device
model = SimpleNN().to(device)

# 3. Prepare Data - Using the MNIST Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the training and test datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 4. Set Up Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 5. Training the Neural Network
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in trainloader:
        # Flatten the input images
        inputs = inputs.view(inputs.shape[0], -1).to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

print("Training complete.")

# 6. Evaluate the Model
correct = 0
total = 0

with torch.no_grad():  # No need to calculate gradients during evaluation
    for inputs, labels in testloader:
        inputs = inputs.view(inputs.shape[0], -1).to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# 7. Save the Model
torch.save(model.state_dict(), 'model.pth')
print("Model saved to 'model.pth'.")

# 8. Load the Model
loaded_model = SimpleNN()
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model = loaded_model.to(device)
print("Model loaded and ready for inference.")
