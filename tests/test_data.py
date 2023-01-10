import torch
from tests import _PATH_DATA
from src.data_script import CorruptMnist


def test_train_data():
     
    dataset = CorruptMnist(train=True)   
    assert len(dataset) == 25000   

    assert dataset[0][0].shape == torch.Size([1,28,28])

#    assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
#    assert that all labels are represented

def test_test_data():
    dataset = CorruptMnist(train=False)   
    assert len(dataset) == 5000   