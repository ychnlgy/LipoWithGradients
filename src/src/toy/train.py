import torch, tqdm
import torch.utils.data

import src

def main(dataset, testset, model, batch, E_main, E_zero, miles_main, miles_zero, optimize_zero):

    dset = torch.utils.data.TensorDataset(*dataset)
    load = torch.utils.data.DataLoader(dset, batch_size=batch, shuffle=True)

    lossf = torch.nn.MSELoss()
    main_optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)

    zero_optim = src.modules.ZeroOptimizer(model.parameters(), lr=0.001)

    extra_optim = [main_optim, zero_optim][optimize_zero]

    for optim, epochs, milestones in zip([main_optim, extra_optim], [E_main, E_zero], [miles_main, miles_zero]):
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones)

        bar = tqdm.tqdm(range(epochs), ncols=80)

        avg = src.util.MovingAverage(momentum=0.95)

        for epoch in bar:

            model.train()
            for X, Y in load:
                Yh = model(X).mean(dim=-1)
                loss = lossf(Yh, Y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                avg.update(loss.item())
                bar.set_description("Loss %.5f" % avg.peek())

            sched.step()

    model.eval()
    with torch.no_grad():
        X_data, Y_data = dataset
        Yh_data = model(X_data).mean(dim=-1)
        data_loss = lossf(Yh_data, Y_data).item()
            
        X_test, Y_test = testset
        Yh_test = model(X_test).mean(dim=-1)
        test_loss = lossf(Yh_test, Y_test).item()

        return data_loss, test_loss
