import torch
from agent.networks import CNN

class BCAgent:
    
    def __init__(self):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize

        return loss

    def predict(self, X):
        # TODO: forward pass
        return outputs

    def load(self, file_name):
        torch.save(self.net.state_dict(), file_name)

    def save(self, file_name):
        self.net.load_state_dict(torch.load(file_name))
