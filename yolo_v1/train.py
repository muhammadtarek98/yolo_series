import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import dataset,compose
from loss import YoloLoss
from yolo_v1 import YoloV1

seed=123
torch.manual_seed(seed)
lr=2e-5
device='cuda' if torch.cuda.is_available() else 'cpu'
batch_size=16
weight_decay=0
transform=compose([transforms.Resize((448,448)),transforms.ToTensor()])

def train(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(predictions=pred, target=y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
    print(f"mean loss {sum(mean_loss) / len(mean_loss)}")


model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss = YoloLoss()
train_dataset = dataset(csv_file='/kaggle/input/pascalvoc-yolo/train.csv',
                        img_dir='/kaggle/input/pascalvoc-yolo/images',
                        label_dir='/kaggle/input/pascalvoc-yolo/labels',
                        s=7, b=2, c=20, transform=transform)
test_dataset = dataset(csv_file='/kaggle/input/pascalvoc-yolo/test.csv',
                       img_dir='/kaggle/input/pascalvoc-yolo/images',

                       label_dir='/kaggle/input/pascalvoc-yolo/labels',
                       s=7, b=2, c=20, transform=transform)
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
for epoch in range(10):
    train(trainloader, model, optimizer, loss)