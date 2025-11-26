import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def train_cls():
    # 1. 数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 加载之前清洗好的 Cropped 数据
    train_data = datasets.ImageFolder('datasets/cls_cropped/train', transform=transform)
    val_data = datasets.ImageFolder('datasets/cls_cropped/val', transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    # 3. 定义模型 (3分类: Nevus, Melanoma, SK)
    model = torchvision.models.mobilenet_v3_large(pretrained=True)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 4. 简单训练循环
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("开始训练分类模型 (Task 3)...")
    for epoch in range(20):  # 练20轮够了
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} 完成")

    # 保存模型
    torch.save(model.state_dict(), 'cls_task3_model.pth')
    print("分类模型已保存！")


if __name__ == '__main__':
    train_cls()