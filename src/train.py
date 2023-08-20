import torch.nn as nn
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class TrainModel(nn.Module):
    def __init__(self, model, criterion, optimizer):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train_one_epoch(self, loader):
        self.model.train()
        acc = 0
        curr_loss = 0
        for graph in tqdm(loader, total=len(loader)):
            x = graph.x.to(device)
            y = graph.y.to(device)
            edge_index = graph.edge_index.to(device)
            batch = graph.batch.to(device)

            self.optimizer.zero_grad()
            out = self.model(x, edge_index, batch)
            # print(out.shape) # batch_size x labels
            # print(y.shape) # batch_size
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            curr_loss += loss.item()
            acc += int((out.argmax(dim=1) == y).sum())

        return curr_loss/len(loader.dataset), acc/len(loader.dataset)

    def test_one_epoch(self, loader):
        self.model.eval()
        acc = 0
        curr_loss = 0
        with torch.no_grad():
            for graph in loader:
                x = graph.x.to(device)
                y = graph.y.to(device)
                edge_index = graph.edge_index.to(device)
                batch = graph.batch.to(device)
                out = self.model(x, edge_index, batch)
                loss = self.criterion(out, y)
                curr_loss += loss.item()
                acc += int((out.argmax(dim=1) == y).sum())
        return curr_loss/len(loader.dataset), acc/len(loader.dataset)
