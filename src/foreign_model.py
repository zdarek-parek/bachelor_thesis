# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://discuss.pytorch.org/t/how-to-change-no-of-input-channels-to-a-pretrained-model/19379

from torchvision import models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
import time
from tempfile import TemporaryDirectory
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

import custom_data_trans as ct
import weights_distr as wd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 25

def change_model_arch(input_4dim_flag:bool):
    '''Loads pretrained pytorch AlexNet model and modifies its architecture, returns modified model.'''
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    if input_4dim_flag:
        pretrained_weights = model.features[0].weight
        new_featres = nn.Sequential(*list(model.features.children()))
        new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
        new_featres[0].weight.data.normal_(0, 0.001)
        new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        model.features = new_featres
    print(model)
    return model

def data_prep(data_dir:str, custom_dataset_flag:bool):
    '''Creates dataset out of the data in the data_dir.
    Returns a dataloader, dataset size and class names.'''
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    } # for datasets.ImageFolder

    if custom_dataset_flag:
        image_datasets = {x: ct.my_dataset(os.path.join(data_dir, x)) for x in ['train', 'val']}
    else: 
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


def get_weight_dist(data_dir:str, class_names):
    '''Computes weights and assigns them to the class names.'''
    dist = wd.util(dataset_path=os.path.join(data_dir,'train'))
    dist_list = []
    for cn in class_names:
        dist_list.append(dist[cn])
    weights = torch.FloatTensor(dist_list)
    return weights

def plot_results(train_acc, val_acc, train_loss, val_loss, epochs):
    '''Saves and displays train and validation loss and accuracy ver the epochs.'''
    figure, axis = plt.subplots(1, 2)

    x = list(range(epochs))

    axis[0].set_title("Training and Validation Loss")
    axis[0].plot(val_loss, label="val")
    axis[0].plot(x, train_loss, label="train")
    # axis[0].set_xticks(x)
    axis[0].set_xlabel("iterations")
    axis[0].set_ylabel("Loss")
    axis[0].legend()

    axis[1].set_title("Training and Validation Accuracy")
    axis[1].plot(val_acc, label="val")
    axis[1].plot(x, train_acc, label="train")
    # axis[1].set_xticks(x)
    axis[1].set_xlabel("iterations")
    axis[1].yaxis.set_label_position("right")
    axis[1].set_ylabel("Accuracy")
    axis[1].legend()
    plt.savefig(r".\history.png")

    # plt.show()
    return

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=EPOCHS):
    '''Trains a model, 
    prints the train and validation statistics, 
    saves the weights of the best performing model.'''
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train': model.train()
                else: model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc)

                elif phase == 'val':
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since

        plot_results(train_accs, val_accs, train_losses, val_losses, num_epochs)
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model

data_dir = r"C:\Users\dasha\Desktop\bakalarka_data\dith_crop_dataset\train_val_data_mock"
def util(weights_flag:bool=False, input_4dim_flag:bool=True, custom_dataset_flag:bool=True):
    '''Utility function that combines dataset creation and model training.'''
    dataloaders, dataset_sizes, class_names = data_prep(data_dir, custom_dataset_flag)

    new_model = change_model_arch(input_4dim_flag)
    new_model = new_model.to(device)

    if (weights_flag):
        weights = get_weight_dist(data_dir, class_names)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model = train_model(new_model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes)
    # torch.save(model.state_dict(), r".\torch_model.pt")

util()