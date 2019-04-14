import torch, tqdm
import torch.utils

import util, modules, datasets
from MovingAverage import MovingAverage

def create_baseline_model(D, C):
    return torch.nn.Sequential(
        torch.nn.Conv2d(D, 32, 3, padding=1),
        modules.ResNet(
            modules.ResBlock(
                block = torch.nn.Sequential(
                    #torch.nn.BatchNorm2d(32),
                    #torch.nn.ReLU(),
                    #torch.nn.Conv2d(32, 32, 3, padding=1),
                    
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 64, 3, padding=1, stride=2) # 32 -> 16
                ),
                shortcut = torch.nn.Conv2d(32, 64, 1, stride=2)
            ),
            modules.ResBlock(
                block = torch.nn.Sequential(
                    #torch.nn.BatchNorm2d(64),
                    #torch.nn.ReLU(),
                    #torch.nn.Conv2d(64, 64, 3, padding=1),
                    
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, 3, padding=1, stride=2) # 16 -> 8
                ),
                shortcut = torch.nn.Conv2d(64, 128, 1, stride=2)
            ),
            modules.ResBlock(
                block = torch.nn.Sequential(
                    #torch.nn.BatchNorm2d(128),
                    #torch.nn.ReLU(),
                    #torch.nn.Conv2d(128, 128, 3, padding=1),
                    
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 256, 3, padding=1, stride=2) # 8 -> 4
                ),
                shortcut = torch.nn.Conv2d(128, 256, 1, stride=2)
            )
        ),
        torch.nn.AvgPool2d(4),
        modules.Reshape(256),
        torch.nn.Linear(256, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, C)
        #modules.PrototypeClassifier(1024, 32),
        #modules.polynomial.Activation(32, n_degree=4),
        #torch.nn.Linear(32, C)
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
    optim = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 200])
    
    epochs = 300
    
    data_avg = MovingAverage(momentum=0.99)
    test_avg = MovingAverage(momentum=0.99)
    
    
        
    for epoch in range(epochs):
        
        model.train()
        
        with tqdm.tqdm(dataloader, ncols=80) as bar:
            for X, Y in bar:
                X = X.to(device)
                Y = Y.to(device)
                
                Yh = model(X)
                loss = lossf(Yh, Y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                data_avg.update(loss.item())
                
                bar.set_description("E%d train loss: %.5f" % (epoch, data_avg.peek()))
            
            sched.step()

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
        
