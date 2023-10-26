from utils import *
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import torch.optim as optim
import torch.nn as nn
import torchvision.models.segmentation.fcn as fcn

def validate(model: nn.Module, dataloader: DataLoader, metrics: SegmentationMetrics, device: str):
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for ipt, tgt in dataloader:
            iptd, tgtd = ipt.to(device), tgt.to(device)
            pred = torch.argmax(model(iptd)['out'], dim=1).cpu()
            metrics.update(pred, tgtd.cpu())
    return metrics

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: str):
    model.train()
    epoch_loss = []
    for ipt, tgt in dataloader:
        iptd, tgtd = ipt.to(device), tgt.to(device)
        optimizer.zero_grad()
        pred = model(iptd)['out']
        loss = criterion(pred, tgtd)
        loss.backward()
        optimizer.step()
        epoch_loss += [loss.item()]
    return model, (sum(epoch_loss) / len(dataloader))

def train(model: nn.Module, dataloader_train: DataLoader, dataloader_val: DataLoader, scheduler: optim.lr_scheduler._LRScheduler,
          optimizer: optim.Optimizer, criterion: nn.Module, metrics: SegmentationMetrics, device: str, epochs: int):
    model.to(device)
    epoch_losses = []
    metric_results = []
    for _ in tqdm(range(epochs)):
        model, loss = train_epoch(model, dataloader_train, optimizer, criterion, device)
        epoch_losses += [loss]
        scheduler.step()
        validate(model, dataloader_val, metrics, device)
        metric_results += [metrics.get_results()]
    save_simple_2d_plot(range(epochs), epoch_losses, title='Loss over epochs', xlabel='Epoch', ylabel='Loss')
    save_simple_2d_plot(range(epochs), [x['Mean IoU'] for x in metric_results], title='mIoU over epochs', xlabel='Epoch', ylabel='mIoU')

def main():
    # Parameters
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--iris_train_subset_size', type=int, required=True)
    parser.add_argument('--iris_val_subset_size', type=int, required=True)
    parser.add_argument('--crop_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(args.seed)

    # Dataset preparation
    transform = ExtCompose([ExtRandomCrop(crop_size=args.crop_size),
                            ExtRandomVerticalFlip(p=0.5),
                            ExtRandomHorizontalFlip(p=0.5),
                            ExtNormalize(mean=[0.485], std=[0.225])])
    iris_dataset = data.CASIAIris(root=args.data_root, transform=transform)
    iris_dataset_indices = list(range(len(iris_dataset)))
    random.shuffle(iris_dataset_indices)
    iris_dataset_indices_train = random.sample(iris_dataset_indices, args.iris_train_subset_size)
    iris_dataset_indices_val = list(set(iris_dataset_indices) - set(iris_dataset_indices_train))[:args.iris_val_subset_size]
    iris_dataset_train = Subset(iris_dataset, iris_dataset_indices_train)
    iris_dataset_val = Subset(iris_dataset, iris_dataset_indices_val)
    iris_dataloader_train = DataLoader(iris_dataset_train, batch_size=args.batch_size, shuffle=True)
    iris_dataloader_val = DataLoader(iris_dataset_val, batch_size=1, shuffle=False)
    
    # Model preparation
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    metrics = SegmentationMetrics(num_classes=2)

    train(model, iris_dataloader_train, iris_dataloader_val, scheduler, optimizer, criterion, metrics, device, epochs=args.epochs)

if __name__ == '__main__':
    main()