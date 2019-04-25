import torch, tqdm, random, numpy, statistics

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
            r2 = torch.rand_like(X)
            I = r1 < self.p
            X[I] = r2[I]*(self.b-self.a)+self.a
        return X

def create_baseline_model(D, C):
    
    d = 64

    sim = src.modules.PrototypeSimilarity(d*4, d)
    act = src.modules.polynomial.Activation(d, n_degree=32)
    pre = torch.nn.Conv2d(d*4, d*4, 3, padding=1)
    post = torch.nn.Conv2d(d, d*8, 1, stride=2)
    
    return torch.nn.Sequential(
        
        torch.nn.Conv2d(D, d, 3, padding=1),
        torch.nn.BatchNorm2d(d),
        
        src.modules.ResNet(

            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(d, d, 3, padding=1),
                    torch.nn.BatchNorm2d(d),
                    
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(d, d, 3, padding=1),
                    torch.nn.BatchNorm2d(d),
                )
            ),

            # 32 -> 16
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(d, d, 3, padding=1),
                    torch.nn.BatchNorm2d(d),

                    torch.nn.ReLU(),
                    torch.nn.Conv2d(d, d*2, 3, padding=1, stride=2),
                    torch.nn.BatchNorm2d(d*2),
                ),
                shortcut = torch.nn.Conv2d(d, d*2, 1, stride=2)
            ),

            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(d*2, d*2, 3, padding=1),
                    torch.nn.BatchNorm2d(d*2),
                    
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(d*2, d*2, 3, padding=1),
                    torch.nn.BatchNorm2d(d*2),
                )
            ),

            # 16 -> 8
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(d*2, d*2, 3, padding=1),
                    torch.nn.BatchNorm2d(d*2),

                    torch.nn.ReLU(),
                    torch.nn.Conv2d(d*2, d*4, 3, padding=1, stride=2),
                    torch.nn.BatchNorm2d(d*4),
                ),
                shortcut = torch.nn.Conv2d(d*2, d*4, 1, stride=2)
            ),

            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(d*4, d*4, 3, padding=1),
                    torch.nn.BatchNorm2d(d*4),
                    
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(d*4, d*4, 3, padding=1),
                    torch.nn.BatchNorm2d(d*4),
                )
            ),

##            # 8 -> 4
##            src.modules.ResBlock(
##                block = torch.nn.Sequential(
##                    torch.nn.ReLU(),
##                    torch.nn.Conv2d(d*4, d*4, 3, padding=1),
##                    torch.nn.BatchNorm2d(d*4),
##
##                    torch.nn.ReLU(),
##                    torch.nn.Conv2d(d*4, d*8, 3, padding=1, stride=2),
##                    torch.nn.BatchNorm2d(d*8),
##                ),
##                shortcut = torch.nn.Conv2d(d*4, d*8, 1, stride=2)
##            ),

            

            # 8 -> 4
            src.modules.ResBlock(
                block = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    pre,
                    torch.nn.BatchNorm2d(d*4),

                    sim,
                    Random(p=0.05, a=-1, b=1),
                    act,
                    
                    torch.nn.Dropout2d(p=0.05),
                    post,
                    torch.nn.BatchNorm2d(d*8),
                ),
                shortcut = torch.nn.Conv2d(d*4, d*8, 1, stride=2),
            )
        ),
        torch.nn.AvgPool2d(4),
        src.modules.Reshape(d*8),
        
        torch.nn.ReLU(),
        torch.nn.Linear(d*8, d*16),
        torch.nn.ReLU(),
        torch.nn.Linear(d*16, C)
        
    ), act, sim, pre, post

#GRAD_ACT = []
#GRAD_SIM = []
#GRAD_PRE = []
#GRAD_PST = []

def _main(cycles, download=0, device="cuda", visualize_relu=0, epochs=300, email=""):

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
    testloader = src.tensortools.dataset.create_loader([test_X, test_Y], batch_size=64)
    
    assert IMAGESIZE == (32, 32)
    
    model, act, sim, pre, post = create_baseline_model(CHANNELS, CLASSES)

    model = torch.nn.DataParallel(model).to(device)

    NUM_VISUAL_ACTIVATIONS = 5
    FIGSIZE = (20, 12)

    if visualize_relu:
        act.visualize_relu(k=NUM_VISUAL_ACTIVATIONS, title="ReLU", figsize=FIGSIZE)
        raise SystemExit

    #act.visualize(k=NUM_VISUAL_ACTIVATIONS, title="Initial state", figsize=FIGSIZE)
    
    lossf = torch.nn.CrossEntropyLoss()
    #optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-6) # 
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    
    data_avg = src.util.MovingAverage(momentum=0.99)
    test_avg = src.util.MovingAverage(momentum=0.99)

    for epoch in range(epochs):

        if cycles > 0 and not epoch % cycles:
            sim.set_visualization_count(NUM_VISUAL_ACTIVATIONS)

        #g_act = []
        #g_sim = []
        #g_pre = []
        #g_pst = []
        
        with tqdm.tqdm(dataloader, ncols=80) as bar:
            
            model.train()
            for X, Y in bar:
                X = random_flip(X)
                X = random_crop(X, padding=4).to(device)
                Y = Y.to(device)
                
                Yh = model(X)
                loss = lossf(Yh, Y)
                optim.zero_grad()
                loss.backward()
                
                #g_act.append(act.weight.grad.norm().item())
                #g_sim.append(sim.weight.grad.norm().item())
                #g_pre.append(pre.weight.grad.norm().item())
                #g_pst.append(post.weight.grad.norm().item())
                rel = sim.weight.grad.norm(dim=-1) > 1e-8
                print(rel.long().sum().item(), torch.numel(rel))
                
                optim.step()
                
                data_avg.update(loss.item())
                
                bar.set_description("E%d train loss: %.5f" % (epoch, data_avg.peek()))

            

            #GRAD_ACT.append(statistics.mean(g_act))
            #GRAD_SIM.append(statistics.mean(g_sim))
            #GRAD_PRE.append(statistics.mean(g_pre))
            #GRAD_PST.append(statistics.mean(g_pst))
            
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
                title = "Epoch %d (%.1f%% test accuracy)" % (epoch, test_avg.peek()*100)
                fname = act.visualize(
                    sim,
                    k=NUM_VISUAL_ACTIVATIONS,
                    title=title,
                    figsize=FIGSIZE
                )

    #act.visualize(k=NUM_VISUAL_ACTIVATIONS, title="Epoch %d" % epochs, figsize=FIGSIZE)

                if service is not None:
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
