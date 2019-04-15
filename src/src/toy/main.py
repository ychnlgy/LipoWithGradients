from . import datasets, train, visual, layers
import src

def get_type(model_type, tier):
    node = getattr(layers, model_type)
    return getattr(node, "Tier_%s" % tier)

def main(
    dataname,
    results_savepath,
    model_type,
    optimize_zero,
    tier,
    D,
    E_main,
    E_zero,
    miles_main,
    miles_zero,
    N = 500,
    noise = 0,
    batch = 16
):
    
    dataset, testset, viewset = datasets.create_from_str(dataname, N, D, noise)
    
    D = dataset[0].size(1)
    Model = get_type(model_type, tier)
    model = Model(D)
    print("Parameters: %d" % src.tensortools.paramcount(model))

    trainloss, testloss = train.main(dataset, testset, model, batch, E_main, E_zero, miles_main, miles_zero, optimize_zero)
    print("Training/testing loss: %.5f/%.5f" % (trainloss, testloss))

    visual.visualize(model, results_savepath, *viewset, testloss, tier)
