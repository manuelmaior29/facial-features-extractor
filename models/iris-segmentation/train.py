from utils import *
from PIL import Image
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
import torchvision.models.segmentation.fcn as fcn
import tqdm

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: str):
    epoch_loss = []
    for ipt, tgt in dataloader:
        iptd, tgtd = ipt.to(device), tgt.to(device)
        optimizer.zero_grad()
        pred = model(iptd)['out']
        loss = criterion(pred, tgtd)
        loss.backward()
        optimizer.step()
        epoch_loss += [loss.item()]
        print(loss.item())
    return model, (epoch_loss / len(dataloader))

def train(model: nn.Module, dataloader: DataLoader, scheduler: optim.lr_scheduler._LRScheduler,
          criterion: nn.Module, device: str, epochs: int):
    model.train()
    model.to(device)
    for _ in range(epochs):
        model, loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(loss)
        scheduler.step()
        
if __name__ == "__main__":
    
    data_root = r'C:\Users\Manuel\Projects\GitHub_Repositories\facial-features-extractor\models\iris-segmentation\data\casia-iris-segmentation'
    transform = ExtCompose([ExtRandomCrop(crop_size=256), 
                            ExtRandomVerticalFlip(p=0.5), 
                            ExtRandomHorizontalFlip(p=0.5),
                            ExtNormalize(mean=[0.485], std=[0.225])])

    iris_dataset_train = data.CASIAIris(root=data_root, transform=transform)
    iris_dataset_train = 
    iris_dataloader_train = DataLoader(iris_dataset_train, batch_size=4, shuffle=False)
    
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train(model, iris_dataloader_train, scheduler, criterion, device, epochs=1)