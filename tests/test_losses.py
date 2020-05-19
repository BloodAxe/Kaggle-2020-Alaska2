import torch

from alaska2.loss import ContrastiveCosineEmbeddingLoss, PairwiseRankingLoss


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
