import torch, tqdm, random

import src, datasets

def random_crop(X, padding):
    N, C, W, H = X.size()
    X = torch.nn.functional.pad(X, [padding]*4, mode="reflect")
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
        src.modules.ResNet(
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(32),
                    #src.modules.PrototypeSimilarity(32, 16),
                    #src.modules.polynomial.Activation(16, n_degree=4),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 64, 3, padding=1, stride=2) # 32 -> 16
                ),
                shortcut = torch.nn.Conv2d(32, 64, 1, stride=2)
            ),
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(64),
                    #src.modules.PrototypeSimilarity(64, 32),
                    #src.modules.polynomial.Activation(32, n_degree=4),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, 3, padding=1, stride=2) # 16 -> 8
                ),
                shortcut = torch.nn.Conv2d(64, 128, 1, stride=2)
            ),
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(128),
                    src.modules.PrototypeSimilarity(128, 32),
                    src.modules.polynomial.Activation(32, n_degree=4),
                    torch.nn.Conv2d(32, 64, 3, padding=1, stride=2) # 8 -> 4
                ),
                shortcut = torch.nn.Conv2d(128, 64, 1, stride=2)
            )
        ),
        torch.nn.AvgPool2d(4),
        src.modules.Reshape(64),
        torch.nn.Linear(64, 256),
        src.modules.PrototypeSimilarity(256, 64),
        src.modules.polynomial.Activation(64, n_degree=4),
        torch.nn.Linear(64, C)
    )

@src.util.main
def main(download=0, device="cuda"):

    download = int(download)
    
    (
        data_X, data_Y, test_X, test_Y, CLASSES, CHANNELS, IMAGESIZE
    ) = datasets.mnist.get(download)
    
    dataloader = src.tensortools.dataset.create_loader([data_X, data_Y], batch_size=32, shuffle=True)
    testloader = src.tensortools.dataset.create_loader([test_X, test_Y], batch_size=128)
    
    assert IMAGESIZE == (32, 32)
    
    model = create_baseline_model(CHANNELS, CLASSES).to(device)
    
    lossf = torch.nn.CrossEntropyLoss()
    #optim = torch.optim.Adam(model.parameters())
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.5, verbose=True)
    
    epochs = 100
    
    data_avg = src.util.MovingAverage(momentum=0.99)
    test_avg = src.util.MovingAverage(momentum=0.99)

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
        
