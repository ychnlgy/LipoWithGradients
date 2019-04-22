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
    act = src.modules.polynomial.Activation(256, n_degree=32)
    return torch.nn.Sequential(
        torch.nn.Conv2d(D, 32, 3, padding=1),
        src.modules.ResNet(
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(),
                    #src.modules.PrototypeSimilarity(32, 32),
                    #src.modules.polynomial.Activation(32, n_degree=3),
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(),
                    #src.modules.PrototypeSimilarity(32, 32),
                    #src.modules.polynomial.Activation(32, n_degree=3),
                    torch.nn.Conv2d(32, 64, 3, padding=1, stride=2) # 32 -> 16
                ),
                shortcut = torch.nn.Sequential(
                    torch.nn.AvgPool2d(2),
                    torch.nn.Conv2d(32, 64, 1)
                )
            ),
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    #src.modules.PrototypeSimilarity(64, 64),
                    #src.modules.polynomial.Activation(64, n_degree=3),
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(),
                    #src.modules.PrototypeSimilarity(64, 32),
                    #src.modules.polynomial.Activation(32, n_degree=3),
                    torch.nn.Conv2d(64, 128, 3, padding=1, stride=2) # 16 -> 8
                ),
                shortcut = torch.nn.Sequential(
                    torch.nn.AvgPool2d(2),
                    torch.nn.Conv2d(64, 128, 1)
                )
            ),
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(128),
                    #torch.nn.ReLU(),
                    src.modules.PrototypeSimilarity(128, 128),
                    src.modules.polynomial.Activation(128, n_degree=16),
                    #torch.nn.Conv2d(128, 128, 3, padding=1),
                    #src.modules.PrototypeSimilarity(128, 64),
                    #src.modules.polynomial.Activation(64, n_degree=4),
                    #torch.nn.Conv2d(64, 128, 3, padding=1),
                    
                    #torch.nn.BatchNorm2d(128),
                    #torch.nn.ReLU(),
##                    src.modules.PrototypeSimilarity(128, 64),
##                    src.modules.polynomial.Activation(64, n_degree=16),
                    torch.nn.Conv2d(128, 256, 3, padding=1, stride=2),
                    #src.modules.PrototypeSimilarity(128, 64),
                    #src.modules.polynomial.Activation(64, n_degree=4),
                    #torch.nn.Conv2d(64, 128, 3, padding=1, stride=2) # 8 -> 4
                ),
                shortcut = torch.nn.Sequential(
                    torch.nn.AvgPool2d(2),
                    torch.nn.Conv2d(128, 256, 1)
                )
            )
        ),
        torch.nn.AvgPool2d(4),
        src.modules.Reshape(256),
        #torch.nn.Linear(128, 256),
        
        #torch.nn.Dropout(p=0.2),
        #torch.nn.ReLU(),
        #torch.nn.Linear(256, C)
        #src.modules.PrototypeSimilarity(256, 64),
        #src.modules.polynomial.Activation(64, n_degree=8),
        #torch.nn.Dropout(p=0.2),
        #torch.nn.ReLU(),
        src.modules.PrototypeSimilarity(256, 256),
        act,
        torch.nn.Linear(256, C)
    ), act

@src.util.main
def main(download=0, device="cuda", visualize_relu=0, gradpenalty=1e-2):

    download = int(download)
    visualize_relu = int(visualize_relu)
    gradpenalty = float(gradpenalty)
    
    (
        data_X, data_Y, test_X, test_Y, CLASSES, CHANNELS, IMAGESIZE
    ) = datasets.cifar.get(download)
    
    dataloader = src.tensortools.dataset.create_loader([data_X, data_Y], batch_size=64, shuffle=True)
    testloader = src.tensortools.dataset.create_loader([test_X, test_Y], batch_size=128)
    
    assert IMAGESIZE == (32, 32)
    
    model, act = create_baseline_model(CHANNELS, CLASSES)
    model = model.to(device)

    NUM_VISUAL_ACTIVATIONS = 5
    FIGSIZE = (20, 6)

    if visualize_relu:
        act.visualize_relu(k=NUM_VISUAL_ACTIVATIONS, title="ReLU", figsize=FIGSIZE)
        raise SystemExit

    act.visualize(k=NUM_VISUAL_ACTIVATIONS, title="Initial state", figsize=FIGSIZE)
    
    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-6)
    #optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.5, verbose=True)
    
    epochs = 400
    
    data_avg = src.util.MovingAverage(momentum=0.99)
    test_avg = src.util.MovingAverage(momentum=0.99)

    for epoch in range(epochs):
        
        model.train()
        
        with tqdm.tqdm(dataloader, ncols=80) as bar:
            for X, Y in bar:
                X = random_crop(X, padding=2).to(device)
                Y = Y.to(device)
                
                Yh = model(X)
                loss = lossf(Yh, Y)# + gradpenalty*src.algorithm.grad_penalty.lipschitz_max_grad(model, X, data_X, data_Y)
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

            if not epoch % 1:
                act.visualize(k=NUM_VISUAL_ACTIVATIONS, title="Epoch %d (%.1f%% test accuracy)" % (epoch, test_avg.peek()*100), figsize=FIGSIZE)
        
    act.visualize(k=NUM_VISUAL_ACTIVATIONS, title="Epoch %d" % epochs, figsize=FIGSIZE)
