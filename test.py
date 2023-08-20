from Model.model import BaseModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model =BaseModel(3,64,32,28).to(device)
model.load_state_dict(torch.load('./Model/base_model.pt'))

