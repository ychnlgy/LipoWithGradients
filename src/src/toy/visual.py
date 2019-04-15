import torch

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot

def visualize(model, outf, X_view, Y_view, testloss, tier):

    with torch.no_grad():

        model.to("cpu")

        yh = model(X_view).mean(dim=-1).numpy()
        y = Y_view.numpy()
        x = X_view[:,0].numpy()
        
        pyplot.plot(x, yh, ".--", label="Predicted trajectory")
        pyplot.plot(x, y,  label="Ground truth")

        pyplot.legend()

        info = "MSE: %.3f; Parameters: %s; Layers: %d" % (testloss, tier, model.count_layers())
        
        pyplot.title(info)
        pyplot.legend()
        pyplot.savefig(outf)
        print("Saved to \"%s\"." % outf)

        pyplot.clf()
