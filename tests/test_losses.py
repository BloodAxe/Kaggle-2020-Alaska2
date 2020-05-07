import torch

from alaska2.loss import ContrastiveCosineEmbeddingLoss


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
