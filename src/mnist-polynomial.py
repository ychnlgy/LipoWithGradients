import torch, tqdm, random, numpy

#import scipy.misc

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

def random_flip(X):
    N = len(X)
    I = torch.rand(N) < 0.5
    numpy.flip(X[I].numpy(), axis=2)
    return X

class PartModel(torch.nn.Module):

    def __init__(self, body, main):
        super().__init__()
        self.body = body
        self.main = main

    def forward(self, X):
        with torch.no_grad():
            X = self.body(X)
        return self.main(X)

    def parameters(self):
        return self.main.parameters()

class Random(torch.nn.Module):

    def __init__(self, p, a, b):
        super().__init__()
        self.p = p
        self.a = a
        self.b = b

    def forward(self, X):
        if self.training:
            r1 = torch.rand_like(X)
            r2 = torch.rand_like(X)
            I = r1 < self.p
            X[I] = r2[I]*(self.b-self.a)+self.a
        return X

def create_baseline_model(D, C):
    act = src.modules.polynomial.Activation(256, n_degree=32)
    sim = src.modules.PrototypeSimilarity(256, 256)
    return torch.nn.Sequential(
        
        torch.nn.Conv2d(D, 32, 3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.LeakyReLU(),
        
        src.modules.ResNet(

            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(),
                ),
                activation = torch.nn.LeakyReLU()
            ),

            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(),
                ),
                activation = torch.nn.LeakyReLU()
            ),
            
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, 3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(32, 64, 3, padding=1, stride=2), # 32 -> 16
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU()
                ),
                shortcut = torch.nn.Conv2d(32, 64, 1, stride=2),
                activation = torch.nn.LeakyReLU()
            ),

            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(),

                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU()
                ),
                activation = torch.nn.LeakyReLU()
            ),

            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU()
                ),
                activation = torch.nn.LeakyReLU()
            ),
            
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU()
                    
                    torch.nn.Conv2d(64, 128, 3, padding=1, stride=2), # 16 -> 8
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU()
                ),
                shortcut = torch.nn.Conv2d(64, 128, 1),
                activation = torch.nn.LeakyReLU()
            ),

            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU()
                ),
                activation = torch.nn.LeakyReLU()
            ),

            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU()
                ),
                activation = torch.nn.LeakyReLU()
            ),
            
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 128, 3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(),
                    
                    torch.nn.Conv2d(128, 256, 3, padding=1, stride=2),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(),
                ),
                shortcut = torch.nn.Conv2d(128, 256, 1),
                activation = torch.nn.LeakyReLU()
            )
        ),
        torch.nn.AvgPool2d(4),
        src.modules.Reshape(256),

        torch.nn.Linear(256, 1024),
        torch.nn.Dropout(p=0.2),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, C)
        
        
        #torch.nn.Linear(256, 256),
        #sim,
        #Random(p=0.05, a=-1, b=1),
        #act,
        #torch.nn.Dropout(p=0.05),
        
        #torch.nn.Linear(256, C)
    ), act, sim

@src.util.main
def main(cycles, download=0, device="cuda", visualize_relu=0, epochs=300):

    download = int(download)
    visualize_relu = int(visualize_relu)
    cycles = int(cycles)
    epochs = int(epochs)
    
    (
        data_X, data_Y, test_X, test_Y, CLASSES, CHANNELS, IMAGESIZE
    ) = datasets.cifar.get(download)

    miu = data_X.mean(dim=0).unsqueeze(0)
    std = data_X.std(dim=0).unsqueeze(0)

    data_X = (data_X-miu)/std
    test_X = (test_X-miu)/std
    
    dataloader = src.tensortools.dataset.create_loader([data_X, data_Y], batch_size=128, shuffle=True)
    testloader = src.tensortools.dataset.create_loader([test_X, test_Y], batch_size=256)
    
    assert IMAGESIZE == (32, 32)
    
    model, act, sim = create_baseline_model(CHANNELS, CLASSES)

    model = model.to(device)

    NUM_VISUAL_ACTIVATIONS = 5
    FIGSIZE = (20, 12)

    if visualize_relu:
        act.visualize_relu(k=NUM_VISUAL_ACTIVATIONS, title="ReLU", figsize=FIGSIZE)
        raise SystemExit

    #act.visualize(k=NUM_VISUAL_ACTIVATIONS, title="Initial state", figsize=FIGSIZE)
    
    lossf = torch.nn.CrossEntropyLoss()
    #optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) #
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    
    data_avg = src.util.MovingAverage(momentum=0.99)
    test_avg = src.util.MovingAverage(momentum=0.99)

    for epoch in range(epochs):

        if cycles > 0 and not epoch % cycles:
            sim.set_visualization_count(NUM_VISUAL_ACTIVATIONS)
        
        with tqdm.tqdm(dataloader, ncols=80) as bar:
            
            model.train()
            for X, Y in bar:
                X = random_flip(X)
                X = random_crop(X, padding=4).to(device)
                Y = Y.to(device)
                
                Yh = model(X)
                loss = lossf(Yh, Y) # + gradpenalty*src.algorithm.grad_penalty.lipschitz_max_grad(model, X, data_X, data_Y)
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

            if cycles > 0 and not epoch % cycles:
                plot = sim.visualize(title="Prototype outputs count", figsize=FIGSIZE)
                act.visualize(plot, k=NUM_VISUAL_ACTIVATIONS, title="Epoch %d (%.1f%% test accuracy)" % (epoch, test_avg.peek()*100), figsize=FIGSIZE)

    #act.visualize(k=NUM_VISUAL_ACTIVATIONS, title="Epoch %d" % epochs, figsize=FIGSIZE)
