import torch

from alaska2.loss import ContrastiveCosineEmbeddingLoss, PairwiseRankingLoss, PairwiseRankingLossV2, RocAucLoss


def test_roc_auc_loss():
    c = RocAucLoss()

    x = torch.tensor([-10, 10]).float()
    y = torch.tensor([0, 1])
    loss = c(x, y)
    print(loss)

    x = torch.tensor([10, -10]).float()
    y = torch.tensor([0, 1])
    loss = c(x, y)
    print(loss)

    x = torch.tensor([0, 0]).float()
    y = torch.tensor([0, 1])
    loss = c(x, y)
    print(loss)

    x = torch.tensor([10, 10]).float()
    y = torch.tensor([0, 1])
    loss = c(x, y)
    print(loss)

    x = torch.tensor([-10, -9]).float()
    y = torch.tensor([0, 1])
    loss = c(x, y)
    print(loss)

    import matplotlib.pyplot as plt
    import numpy as np

    x_coords = np.arange(-5, 5, 0.1)
    y_loss_neg = []
    y_loss_pos = []
    for x_val in x_coords:
        x = torch.tensor([x_val, -2], dtype=torch.float32)
        y = torch.tensor([0, 1])
        loss = c(x, y)
        y_loss_neg.append(loss.item())

        x = torch.tensor([+2, x_val], dtype=torch.float32)
        y = torch.tensor([0, 1])
        loss = c(x, y)
        y_loss_pos.append(loss.item())

    plt.figure()
    plt.plot(x_coords, y_loss_neg, label="neg")
    plt.plot(x_coords, y_loss_pos, label="pos")
    plt.legend()
    plt.show()


def test_pairwise_rank_loss():
    c = PairwiseRankingLoss()

    x = torch.tensor([-10, 10]).float()
    y = torch.tensor([0, 1])
    loss = c(x, y)
    print(loss)

    x = torch.tensor([10, -10]).float()
    y = torch.tensor([0, 1])
    loss = c(x, y)
    print(loss)

    x = torch.tensor([0, 0]).float()
    y = torch.tensor([0, 1])
    loss = c(x, y)
    print(loss)

    x = torch.tensor([10, 10]).float()
    y = torch.tensor([0, 1])
    loss = c(x, y)
    print(loss)

    x = torch.tensor([-10, -9]).float()
    y = torch.tensor([0, 1])
    loss = c(x, y)
    print(loss)


def test_pairwise_rank_loss_v2():
    c = PairwiseRankingLossV2()

    # x = torch.tensor([-10, 10, -10, 10]).float()
    y = torch.tensor([0, 1, 0, 1])
    # loss = c(x, y)
    # print(loss)
    #
    # x = torch.tensor([10, -10, -10, 10]).float()
    # loss = c(x, y)
    # print(loss)
    #
    # x = torch.tensor([0, 0, 0, 0]).float()
    # loss = c(x, y)
    # print(loss)
    #
    # x = torch.tensor([10, 10, -10, -10]).float()
    # loss = c(x, y)
    # print(loss)

    x = torch.tensor([-10, -1, 1, 10]).float()
    # x = torch.tensor([-10, 0, 0, 10]).float()
    loss = c(x, y)
    print(loss)


def test_ccos():
    # input = torch.ones((8, 32)).float()
    # target = torch.zeros(8).long()
    #
    # criterion = ContrastiveCosineEmbeddingLoss()
    # loss = criterion(input, target)
    # print(loss)
    # assert loss == 0
    #
    # input = torch.ones((8, 32)).float()
    # target = torch.ones(8).long()
    #
    # criterion = ContrastiveCosineEmbeddingLoss()
    # loss = criterion(input, target)
    # print(loss)
    # assert loss == 0

    input = torch.randn((160, 32)).float().cuda()
    target = torch.randint(4, (160,)).long().cuda()

    criterion = ContrastiveCosineEmbeddingLoss()
    loss = criterion(input, target)
    print(loss)
