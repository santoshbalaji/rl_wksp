import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


class Model(nn.Module):

    def __init__(
            self,
            in_features = 4,
            h1 = 3,
            h2 = 3,
            out_features = 3):
        super().__init__()
        self.__fc1 = nn.Linear(in_features, h1)
        self.__fc2 = nn.Linear(h1, h2)
        self.__out = nn.Linear(h2, out_features)


    def forward(self, x):
        x = F.relu(self.__fc1(x))
        x = F.relu(self.__fc2(x))
        x = self.__out(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(40)
    model = Model()

    url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    data = pd.read_csv(url)
    data['variety'] = data['variety'].replace('Setosa', 0.0)
    data['variety'] = data['variety'].replace('Versicolor', 1.0)
    data['variety'] = data['variety'].replace('Virginica', 2.0)

    X = data.drop('variety', axis=1)
    y = data['variety']

    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    epochs = 400
    losses = []

    for i in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)

        losses.append(loss.detach().numpy())

        if i % 10 == 0:
            print(f'Epoch: {i} and loss: {loss}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(range(epochs), losses)
    plt.ylabel('Loss/Error')
    plt.xlabel('Epoch')
    plt.show()

    correct = 0
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_val = model.forward(data)

            if y_test[i] == 0:
                x = "Setosa"
            elif y_test[i] == 1:
                x = "Versicolor"
            elif y_test[i] == 2:
                x = "Virginica"
            
            print(f'{i + 1}.) {str(y_val)} \t {x} \t {y_val.argmax().item()}')

            if y_val.argmax().item() == y_test[i]:
                correct = correct + 1
    
    print(f'We got {correct} correct!')

    torch.save(model.state_dict(), 'test_nn.pt')
    
    new_model = Model()
    new_model.load_state_dict(torch.load(
        'test_nn.pt', 
        weights_only=False))
    
    print(new_model)
    print(new_model.parameters())
