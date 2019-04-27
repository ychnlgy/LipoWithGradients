import torch, tqdm, random, numpy, statistics

#import scipy.misc

import src, datasets

def random_crop(X, padding):
    N, C, W, H = X.size()
    X = torch.nn.functional.pad(X, [padding]*4, mode="constant")
    out = [
        _random_crop(X[i], padding, W, H) for i in range(N)
    ]
    return torch.stack(out, dim=0)

def random_cutout(X, l):
    N, C, W, H = X.size()
    out = [
        _random_cutout(X[i], l, W, H) for i in range(N)
    ]
    return torch.stack(out, dim=0)

def _random_cutout(x, l, W, H):
    xi = random.randint(-l+1, W-1)
    yi = random.randint(-l+1, H-1)
    xj = xi + l
    yj = yi + l
    xi = max(0, xi)
    yi = max(0, yi)
    x = x.clone()
    x[:,xi:xj,yi:yj] = 0
    return x

def _random_crop(x, padding, W, H):
    w_i = random.randint(0, padding)
    h_i = random.randint(0, padding)
    return x[:,w_i:w_i+W,h_i:h_i+H]

def random_flip(X):
    #X = _random_flip(X, axis=2) # vertical
    X = _random_flip(X, axis=3) # horizontal
    return X

def _random_flip(X, axis):
    N = len(X)
    I = torch.rand(N) < 0.5
    X[I] = torch.from_numpy(numpy.flip(X[I].numpy(), axis=axis).copy())
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
            r2 = torch.rand_like(X)*(self.b-self.a)+self.a
            I = (r1 < self.p).float()
            X = X*(1-I) + I*r2
        return X

def create_midact(d, act=None):
    act = act if act is not None else src.modules.polynomial.Activation(d, n_degree=3)
    return torch.nn.Sequential(
        torch.nn.Tanh(),
        Random(p=0.05, a=-1, b=1),
        act,
        torch.nn.Dropout2d(p=0.05),
        #torch.nn.ReLU()
    )
        #torch.nn.ReLU()

def create_baseblock(d):
    return src.modules.ResBlock(
        block = torch.nn.Sequential(
            torch.nn.ReLU(),
            #create_midact(d),
            torch.nn.Conv2d(d, d, 3, padding=1),
            torch.nn.BatchNorm2d(d),
            
            create_midact(d),
            torch.nn.Conv2d(d, d, 3, padding=1),
            torch.nn.BatchNorm2d(d),
        )
    )

def create_skipblock(d, act):
    return src.modules.ResBlock(
        block = torch.nn.Sequential(
            torch.nn.ReLU(),
            #create_midact(d),
            torch.nn.Conv2d(d, d, 3, padding=1),
            torch.nn.BatchNorm2d(d),

            create_midact(d, act),
            torch.nn.Conv2d(d, d*2,  3, padding=1, stride=2),
            torch.nn.BatchNorm2d(d*2),
        ),
        shortcut = torch.nn.Conv2d(d, d*2, 1, stride=2)
    )

#def create_polyblock(d):
    

def create_baseline_model(D, C):
    
    d = 32
    act = src.modules.polynomial.Activation(d*4, n_degree=3)
    
    return torch.nn.Sequential(
        
        torch.nn.Conv2d(D, d, 3, padding=1),
        torch.nn.BatchNorm2d(d),
        
        src.modules.ResNet(
            #create_baseblock(d),
            create_baseblock(d),
            create_skipblock(d), # 32 -> 16
            #create_baseblock(d*2),
            create_baseblock(d*2),
            create_skipblock(d*2), # 16 -> 8
            #create_baseblock(d*4),
            create_baseblock(d*4),
            create_skipblock(d*4, act) # 8 -> 4
        ),
        torch.nn.AvgPool2d(4),
        src.modules.Reshape(d*8),
        
        torch.nn.ReLU(),
        torch.nn.Linear(d*8, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, C)
        
    ), act

#GRAD_ACT = []
#GRAD_SIM = []
#GRAD_PRE = []
#GRAD_PST = []

def _main(cycles, download=0, device="cuda", visualize_relu=0, epochs=150, email=""):

    download = int(download)
    visualize_relu = int(visualize_relu)
    cycles = int(cycles)
    epochs = int(epochs)

    service = None
    if email:
        import mailupdater
        service = mailupdater.Service(email)
    
    (
        data_X, data_Y, test_X, test_Y, CLASSES, CHANNELS, IMAGESIZE
    ) = datasets.cifar.get(download)

    miu = data_X.mean(dim=0).unsqueeze(0)

    data_X = (data_X-miu)
    test_X = (test_X-miu)
    
    dataloader = src.tensortools.dataset.create_loader([data_X, data_Y], batch_size=32, shuffle=True)
    testloader = src.tensortools.dataset.create_loader([test_X, test_Y], batch_size=128)
    
    assert IMAGESIZE == (32, 32)
    
    model, act = create_baseline_model(CHANNELS, CLASSES)

    num_params = sum(torch.numel(p) for p in model.parameters() if p.requires_grad)
    print("Parameters: %d" % num_params)

    model = torch.nn.DataParallel(model).to(device)

    NUM_VISUAL_ACTIVATIONS = 5
    FIGSIZE = (20, 6)
    
    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-6) # 
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    
    data_avg = src.util.MovingAverage(momentum=0.99)

    for epoch in range(1, epochs+1):
        
        with tqdm.tqdm(dataloader, ncols=80) as bar:
            
            model.train()
            for X, Y in bar:
                X = random_flip(X)
                X = random_crop(X, padding=4)
                X = random_cutout(X, l=8)
                X = X.to(device)
                Y = Y.to(device)
                
                Yh = model(X)
                loss = lossf(Yh, Y)
                optim.zero_grad()
                loss.backward()
                
                optim.step()
                
                data_avg.update(loss.item())
                
                bar.set_description("E%d train loss: %.5f" % (epoch, data_avg.peek()))
            
            sched.step(data_avg.peek())


            test_acc = test_n = 0.0
            
            model.eval()
            with torch.no_grad():
                for X, Y in testloader:
                    X = X.to(device)
                    Y = Y.to(device)
                    Yh = model(X)
                    match = Yh.max(dim=1)[1] == Y
                    acc = match.float().sum()
                    
                    test_acc += acc.item()
                    test_n += torch.numel(match)

            test_avg = test_acc/test_n

            print("Test accuracy: %.5f" % test_avg)

            if not epoch % cycles:
                fname = act.visualize(k=NUM_VISUAL_ACTIVATIONS, title="Epoch %03d" % epochs, figsize=FIGSIZE)
                if service is not None:
                    title = "Epoch %d (%.1f%% test accuracy)" % (epoch, test_avg*100)
                    try:
                        with service.create(title) as email:
                            email.attach(fname)
                    except:
                        pass

def plot_grads():
    import matplotlib
    matplotlib.use("agg")

    fig, axes = matplotlib.pyplot.subplots(nrows=4, sharex=True, figsize=(12, 8))

    for i, (title, y) in enumerate([
        ("Pre-activation", GRAD_PRE),
        ("Prototype cosine-similarity", GRAD_SIM),
        ("Polynomial weights", GRAD_ACT),
        ("Post-activation", GRAD_PST)
    ]):
        axes[i].set_ylabel(title)
        axes[i].plot(y)

    axes[0].set_title("Gradient norm across training epochs")
    axes[-1].set_xlabel("Epochs")

    matplotlib.pyplot.savefig("grads.png", bbox_inches="tight")

@src.util.main
def main(plotg=0, **kwargs):
    plotg = int(plotg)
    try:
        _main(**kwargs)
    except:
        raise
    finally:
        if plotg:
            plot_grads()
