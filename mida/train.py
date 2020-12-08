import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MIDA import Mida
from data_processor import DataProcessor
from sklearn.metrics import mean_squared_error


def train(data_path, num_of_epochs=500):
    data_pros = DataProcessor(data_path)
    _, n_cols, train, test = data_pros.load_data()
    
    missed_test, mask = data_pros.get_missed_data(test, missing_mechanism="MNAR", randomness="random")
    
    missed_test = torch.from_numpy(missed_test).float() #pylint: disable=no-member
    train = torch.from_numpy(train).float() #pylint: disable=no-member

    train_dataLoader = DataLoader(dataset=train, batch_size=1)

    device = torch.device("cpu") #pylint: disable=no-member
    model = Mida(n = train.shape[1]).to(device)

    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.99, lr=0.01, nesterov=True)

    costs = []
    # num_of_batches = len(train)
    stop = False

    for epoch in range(num_of_epochs):
        cost = None
        for i, batch in enumerate(train_dataLoader):
            batch = batch.to(device)
            cost = loss(model(batch), batch)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if cost.item() < 1e-6:
                print("early stop reached")
                stop = True
                break

            costs.append(cost.item())
        
        print(f"Epoch {epoch + 1}, Loss: {cost.item()}")

        if stop:
            break
    
    # evaluation
    model.eval()
    imputed = model(missed_test.to(device))
    imputed = imputed.cpu().detach().numpy()

    total_rmse = 0
    for i in range(n_cols):
        if mask[:, i].sum() > 0:
            actual = test[:, i][mask[:, i]]
            pred = imputed[:, i][mask[:, i]]

            total_rmse += math.sqrt(mean_squared_error(actual, pred))

    print(f"Result: {total_rmse}")

    
if __name__ == "__main__":
    train("data/GL.csv")
