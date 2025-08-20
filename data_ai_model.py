# Creating the Data and the neural network
# Displayinf the graphic to show me how the model is evolvong with the process
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def ai_model():
    # Setting the data to train my model
    x_train  = torch.unsqueeze(torch.linspace(-20,20,100) , dim=1)
    # Now normalizing the Data
    u = x_train.mean()
    std = x_train.std()
    x_train = ( x_train - u ) / std

    # Target NonLinear-Equation
    y = torch.cos(x_train) + x_train.pow(2)
    # Normalizing the Data
    y = ( y - torch.mean(y))/ torch.std(y)

    # Creating the AI model
    model = nn.Sequential(
        nn.Linear(1,32), # Adding the first layer
        nn.ReLU(),# Neuron activation function
        nn.Linear(32,32), # Hidden layer
        nn.ReLU(), # Neuron activation function
        nn.Linear(32,1) # Third and final layer
    )

    # Creating the optimizer based upon the loss from the model's accuracy
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),0.03)

    # Training the AI model
    epochs = 500 # This is the number of times I will set my neural network to be trained
    for i in range (epochs+1):
        y_pred = model(x_train)
        loss = criterion(y_pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0: print(f'Epoch{i},Loss:{loss.item()}') # Tracking down the loss rate from the current AI model.


    # Displaying the graphic

    plt.figure(figsize = (10,5) )
    plt.plot(x_train.numpy(),y.numpy(), label = 'Original', color = 'red')
    plt.plot(x_train.numpy(),y_pred.detach().numpy(), label = 'Predicted' , color = 'green')
    plt.legend()
    plt.title('Non Linear Equation Learning')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

    # Saving the ai model
    
    torch.save(model.state_dict(),'NonLinear-Model.pth')

ai_model()