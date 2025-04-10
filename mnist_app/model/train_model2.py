import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# Define the model
# Starting with a multi-layer perceptron (MLP)
# model = nn.Sequential(
#     nn.Linear(28 * 28, 64),
#     nn.ReLU(),
#     nn.Linear(64, 64),
#     nn.ReLU(),
#     nn.Dropout(0.1), # Dropout layer to prevent overfitting
#     nn.Linear(64, 10)
# )

# # Define a more flexible model
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logits = self.l3(do)
        return logits

model = ResNet()


# Define optimiser
params = model.parameters()
optimiser = optim.SGD(params, lr=0.01)

# Define loss function
loss = nn.CrossEntropyLoss()

# Define training loop
nb_epochs = 5

# Train, val split
train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)


# My training and validatio loops
for epoch in range(nb_epochs):
    losses = list()
    for batch in train_loader:
        x,y = batch
        b = x.size(0)
        x = x.view(b, -1)
        
        # 1) Forward pass
        l = model(x) # l: logits
        l = loss(l, y)
        
        # 2) Compute the objective function 
        J = loss(l, y)
        
        # 3) Cleaning the gradients 
        model.zero_grad()
        
        # 4) Accumulate the partial derivatives of the loss function with respect to the parameters
        J.backward()
        
        # 5) Step in the opposite direction of the gradient
        optimiser.step() #or alternatively with torch.no_grad(): params = params - eta * params.grad
        
        losses.append(J.item())
        
    print(f"Epoch {epoch + 1}, Training Loss: {torch.tensor(losses).mean(): .2f}")
        


    losses = list()
    for batch in val_loader:
        x,y = batch
        b = x.size(0)
        x = x.view(b, -1)
        
        # 1) Forward pass
        with torch.no_grad():
            l = model(x) # l: logits

        
        # 2) Compute the objective function 
        J = loss(l, y)
        losses.append(J.item())
        
    print(f"Epoch {epoch + 1}, Validation Loss: {torch.tensor(losses).mean(): .2f}")
