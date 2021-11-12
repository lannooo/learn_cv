import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lenet5_model import LeNet5
from tiny_alexnet_model import MiniAlexNet

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

learning_rate = 0.001
batch_size = 32
n_epoch = 48
image_size = 32
n_classes = 10


def train(dataloader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct_pred = 0

    for X, y in dataloader:
        optimizer.zero_grad()

        X = X.to(device)
        y = y.to(device)

        y_hat, y_prob = model(X)
        loss = criterion(y_hat, y)
        running_loss += loss.item() * X.size(0)
        label_pred = torch.argmax(y_prob, dim=1)
        correct_pred += (label_pred == y).sum()

        loss.backward()
        optimizer.step()
    total_n = len(dataloader.dataset)
    epoch_loss = running_loss / total_n
    accuracy = correct_pred.float().item() / total_n
    return model, optimizer, epoch_loss, accuracy


def validate(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0
    correct_pred = 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_hat, y_prob = model(X)
        loss = criterion(y_hat, y)
        running_loss += loss.item() * X.size(0)
        label_pred = torch.argmax(y_prob, 1)
        correct_pred += (label_pred == y).sum()

    total_n = len(dataloader.dataset)
    epoch_loss = running_loss / total_n
    accuracy = correct_pred.float().item() / len(dataloader.dataset)
    return model, epoch_loss, accuracy


def train_loop(model, criterion, optimizer, lr_scheduler, dataloader_train, dataloader_valid, epochs, device):
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    for epoch in range(0, epochs):
        model, optimizer, loss_train, accuracy_train = train(dataloader_train, model, criterion, optimizer, device)
        train_losses.append(loss_train)
        train_accs.append(accuracy_train)

        with torch.no_grad():
            model, loss_valid, accuracy_valid = validate(dataloader_valid, model, criterion, device)
            valid_losses.append(loss_valid)
            valid_accs.append(accuracy_valid)
        print(f'epoch: {epoch}, train loss: {loss_train:.4f}, valid loss: {loss_valid:.4f}, train accuracy: {accuracy_train:.4f}, valid accuracy: {accuracy_valid:.4f}')
        lr_scheduler.step()
    return model, optimizer, (train_losses, valid_losses), (train_accs, valid_accs)

def plot_loss_acc(train_losses, valid_losses, train_accs, valid_accs):
    plt.style.use('seaborn')

    fig, ax = plt.subplots(2, 1, figsize = (8, 12))
    ax[0].plot(train_losses, color='blue', label='Training loss')
    ax[0].plot(valid_losses, color='yellow', label='Validation loss')
    ax[0].set(title='Loss over epochs', xlabel='epoch', ylabel='loss',)
    ax[0].legend()
    ax[1].plot(train_accs, color='blue', label='Training accuracy')
    ax[1].plot(valid_accs, color='yellow', label='Validation accuracy')
    ax[1].set(title='accuracy over epochs', xlabel='epoch', ylabel='accuracy',)
    ax[1].legend()

    fig.savefig('accuracy.png')
    fig.show()
    plt.style.use('default')

def plot_prediction(valid_dataset, model):
    ROW_IMG = 10
    N_ROWS = 5
    model.to('cpu')
    fig = plt.figure(figsize=(16, 8))
    classes = valid_dataset.classes
    for index in range(1, ROW_IMG * N_ROWS + 1):
        plt.subplot(N_ROWS, ROW_IMG, index)
        plt.axis('off')
        plt.imshow(valid_dataset.data[index])
        
        with torch.no_grad():
            model.eval()
            _, probs = model(valid_dataset[index][0].unsqueeze(0))
            
        title = f'{classes[torch.argmax(probs).item()]} ({torch.max(probs * 100):.2f}%)'
        
        plt.title(title, fontsize=7)
    fig.suptitle('predictions')
    fig.savefig('prediction.png')
    fig.show()

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    task = 'mnist'
    # task = 'cifar10'
    if task == 'mnist':
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
        valid_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform)
        model = LeNet5(n_classes).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        train_dataset = datasets.CIFAR10(root='cifar10_data', train=True, transform=transform, download=True)
        valid_dataset = datasets.CIFAR10(root='cifar10_data', train=False, transform=transform)
        model = MiniAlexNet(n_classes).to(DEVICE)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.1)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    model, optimizer, losses, accs = train_loop(model, criterion, optimizer, lr_scheduler, train_loader, valid_loader, n_epoch, DEVICE)
    
    train_losses, valid_losses = losses
    train_accs, valid_accs = accs
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)
    train_accs = np.array(train_accs)
    valid_accs = np.array(valid_accs)
    print(f'train loss reseult > min: {train_losses.min()} epoch: {train_losses.argmin()}')
    print(f'valid loss reseult > min: {valid_losses.min()} epoch: {valid_losses.argmin()}')
    print(f'train accuracy reseult > max: {train_accs.max()} epoch: {train_accs.argmax()}')
    print(f'valid accuracy reseult > max: {valid_accs.max()} epoch: {valid_accs.argmax()}')
    plot_loss_acc(train_losses, valid_losses, train_accs, valid_accs)
    plot_prediction(valid_dataset, model)
