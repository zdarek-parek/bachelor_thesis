from torchvision import models
import torch.nn as nn

def model1():    
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    return model

def model2():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)
    return model

def model3():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    return model


def model4():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)

    return model

def model5():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 2048, bias=True)
    model.classifier[-1] = nn.Linear(2048, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)

    return model

def model6():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 2048, bias=True)
    model.classifier[-1] = nn.Linear(2048, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)

    return model

def model7():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(9216, 2048, bias = True)
    model.classifier[4] = nn.Linear(2048, 2048, bias=True)
    model.classifier[-1] = nn.Linear(2048, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    return model

def model8():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(9216, 2048, bias = True)
    model.classifier[4] = nn.Linear(2048, 2048, bias=True)
    model.classifier[-1] = nn.Linear(2048, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)

    return model

def model9():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(9216, 1024, bias = True)
    model.classifier[4] = nn.Linear(1024, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres
    
    return model

def model10():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(9216, 1024, bias = True)
    model.classifier[4] = nn.Linear(1024, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)
    
    return model


def model11(): 
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)   
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=2, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)

    model.features = new_featres
    return model

def model12():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=5, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)

    model.features = new_featres
    return model

def model13():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights).data[:, :3, :7, :7]

    model.features = new_featres
    return model


def model14():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights).data[:, :3, :7, :7]

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)
    
    model.features = new_featres
    return model

def model15():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=15, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :11, :11] = nn.Parameter(pretrained_weights)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)
    
    model.features = new_featres
    return model

def model16():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

    pretrained_weights2 = model.features[3].weight
    new_featres[3] = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=2)
    new_featres[3].weight.data.normal_(0, 0.001)
    new_featres[3].weight.data[:, :, :, :] = nn.Parameter(pretrained_weights2).data[:, :, :3, :3]

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)

    model.features = new_featres
    return model

def model17():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=2, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

    pretrained_weights2 = model.features[3].weight
    new_featres[3] = nn.Conv2d(64, 192, kernel_size=7, stride=2, padding=2)
    new_featres[3].weight.data.normal_(0, 0.001)
    new_featres[3].weight.data[:, :, :5, :5] = nn.Parameter(pretrained_weights2)

    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)

    model.features = new_featres
    return model

def model18():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

    pretrained_weights2 = model.features[3].weight
    new_featres[3] = nn.Conv2d(64, 192, kernel_size=7, stride=2, padding=2)
    new_featres[3].weight.data.normal_(0, 0.001)
    new_featres[3].weight.data[:, :, :5, :5] = nn.Parameter(pretrained_weights2)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)

    model.features = new_featres
    return model

def model19():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=15, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :11, :11] = nn.Parameter(pretrained_weights)

    new_featres[2] = nn.MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)
    
    model.features = new_featres
    return model

def model20():
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :11, :11] = nn.Parameter(pretrained_weights)

    new_featres[2] = nn.MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)

    pretrained_weights2 = model.features[3].weight
    new_featres[3] = nn.Conv2d(64, 192, kernel_size=7, stride=2, padding=2)
    new_featres[3].weight.data.normal_(0, 0.001)
    new_featres[3].weight.data[:, :, :5, :5] = nn.Parameter(pretrained_weights2)

    new_featres[5] = nn.MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)
    
    model.features = new_featres
    return model


def model21(): # 20, inverse weights
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :11, :11] = nn.Parameter(pretrained_weights)

    new_featres[2] = nn.MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)

    pretrained_weights2 = model.features[3].weight
    new_featres[3] = nn.Conv2d(64, 192, kernel_size=7, stride=2, padding=2)
    new_featres[3].weight.data.normal_(0, 0.001)
    new_featres[3].weight.data[:, :, :5, :5] = nn.Parameter(pretrained_weights2)

    new_featres[5] = nn.MaxPool2d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)
    
    model.features = new_featres   
    return model

def model22(): # 11, inverse weights
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=2, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)

    model.features = new_featres
    return model

def model23(): # 12, inv weights
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=5, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)

    model.features = new_featres
    return model

def model24(): # 1, inv weights
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(4096, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres
    return model

def model25(): # 14, inv weights
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights).data[:, :3, :7, :7]

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)
    
    model.features = new_featres
    return model

def model26(): # 15, inv weights
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=15, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :11, :11] = nn.Parameter(pretrained_weights)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)
    
    model.features = new_featres
    return model

def model27(): # 10, inv weights
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(9216, 1024, bias = True)
    model.classifier[4] = nn.Linear(1024, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)

    return model

def model28(): # 6, inv weights
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 2048, bias=True)
    model.classifier[-1] = nn.Linear(2048, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)

    return model

def model29(): # 8 (stride=2), inv weights
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(9216, 2048, bias = True)
    model.classifier[4] = nn.Linear(2048, 2048, bias=True)
    model.classifier[-1] = nn.Linear(2048, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=2, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
    model.features = new_featres

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)
    return model

def model30(): # 18, inv weights
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[4] = nn.Linear(4096, 1024, bias=True)
    model.classifier[-1] = nn.Linear(1024, 17, bias=True)

    pretrained_weights = model.features[0].weight
    new_featres = nn.Sequential(*list(model.features.children()))
    new_featres[0] = nn.Conv2d(4, 64, kernel_size=11, stride=2, padding=2)
    new_featres[0].weight.data.normal_(0, 0.001)
    new_featres[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)

    pretrained_weights2 = model.features[3].weight
    new_featres[3] = nn.Conv2d(64, 192, kernel_size=7, stride=2, padding=2)
    new_featres[3].weight.data.normal_(0, 0.001)
    new_featres[3].weight.data[:, :, :5, :5] = nn.Parameter(pretrained_weights2)

    model.classifier[0] = nn.Dropout(p=0.25, inplace=False)
    model.classifier[3] = nn.Dropout(p=0.25, inplace=False)

    model.features = new_featres
    return model
