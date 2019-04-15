import torch, tqdm, random
import torch.utils

import util, modules, datasets
from MovingAverage import MovingAverage

def random_crop(X, padding):
    N, C, W, H = X.size()
    X = torch.nn.functional.pad(X, [padding]*4)
    out = [
        _random_crop(X[i], padding, W, H) for i in range(N)
    ]
    return torch.stack(out, dim=0)

def _random_crop(x, padding, W, H):
    w_i = random.randint(0, padding)
    h_i = random.randint(0, padding)
    return x[:,w_i:w_i+W,h_i:h_i+H]

def create_baseline_model(D, C):
    return torch.nn.Sequential(
        torch.nn.Conv2d(D, 32, 3, padding=1),
        modules.ResNet(
            modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 64, 3, padding=1, stride=2) # 32 -> 16
                ),
                shortcut = torch.nn.Conv2d(32, 64, 1, stride=2)
            ),
            modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, 3, padding=1, stride=2) # 16 -> 8
                ),
                shortcut = torch.nn.Conv2d(64, 128, 1, stride=2)
            ),
            modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    
                    #torch.nn.BatchNorm2d(128),
                    #torch.nn.ReLU(),
                    #torch.nn.Conv2d(128, 256, 3, padding=1, stride=2)
                    
                    modules.Transpose(1, 3),
                    modules.PrototypeClassifier(128, 32),
                    modules.polynomial.Activation(32, n_degree=4),
                    modules.Transpose(3, 1),
                    torch.nn.Conv2d(32, 256, 3, padding=1, stride=2) # 8 -> 4
                ),
                shortcut = torch.nn.Conv2d(128, 256, 1, stride=2)
            )
        ),
        torch.nn.AvgPool2d(4),
        modules.Reshape(256),
        torch.nn.Linear(256, 1024),
        #torch.nn.Dropout(p=0.4),
        #torch.nn.ReLU(),
        #torch.nn.Linear(1024, C)
        modules.PrototypeClassifier(1024, 32),
        modules.polynomial.Activation(32, n_degree=4),
        torch.nn.Linear(32, C)
    )

#def create_baseline2(D, C):
#    return torch.nn.Sequential(
#        torch.nn.Conv2d(D, 32, 

@util.main
def main(download=0, device="cuda"):

    download = int(download)
    
    (
        data_X, data_Y, test_X, test_Y, CLASSES, CHANNELS, IMAGESIZE
    ) = datasets.mnist.get(download)
    
    dataset = torch.utils.data.TensorDataset(data_X, data_Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    testset = torch.utils.data.TensorDataset(test_X, test_Y)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128)
    
    assert IMAGESIZE == (32, 32)
    
    model = create_baseline_model(CHANNELS, CLASSES).to(device)
    
    lossf = torch.nn.CrossEntropyLoss()
    #optim = torch.optim.Adam(model.parameters())
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.5, verbose=True)
    
    epochs = 100
    
    data_avg = MovingAverage(momentum=0.99)
    test_avg = MovingAverage(momentum=0.99)
    
    
        
    for epoch in range(epochs):
        
        model.train()
        
        with tqdm.tqdm(dataloader, ncols=80) as bar:
            for X, Y in bar:
                X = random_crop(X, padding=2).to(device)
                Y = Y.to(device)
                
                Yh = model(X)
                loss = lossf(Yh, Y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                data_avg.update(loss.item())
                
                bar.set_description("E%d train loss: %.5f" % (epoch, data_avg.peek()))
            
            sched.step(data_avg.peek())

            model.eval()
            with torch.no_grad():
                for X, Y in testloader:
                    X = X.to(device)
                    Y = Y.to(device)
                    Yh = model(X)
                    match = Yh.max(dim=1)[1] == Y
                    acc = match.float().mean()
                    
                    test_avg.update(acc.item())
                
            print("Test accuracy: %.5f" % test_avg.peek())
        
