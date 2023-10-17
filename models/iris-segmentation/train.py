from utils import *
from PIL import Image
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.models.segmentation.fcn as fcn

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: str):
    model.train()
    for ipt, tgt in dataloader:
        ipt, tgt = ipt.to(device), tgt.to(device)
        optimizer.zero_grad()
        pred = model(ipt)
        loss = criterion(pred, tgt)
        loss.backward()
        optimizer.step()
    return model, loss

if __name__ == "__main__":
    data_root = r'C:\Users\Manuel\Projects\GitHub_Repositories\facial-features-extractor\models\iris-segmentation\data\casia-iris-segmentation'
    iris_dataset_train = data.CASIAIris(root=data_root, transform=ExtToTensor())
    iris_dataloader_train = DataLoader(iris_dataset_train, batch_size=4, shuffle=False)
    
    model = torchvision.models.segmentation.deeplabv3_resnet101()
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_epoch(model, iris_dataloader_train, optimizer, criterion, device)