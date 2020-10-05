from torch.utils.data import DataLoader
import torch


def mean_std(in_data_set, exists_mean_std=None):
    if exists_mean_std is not None:
        return exists_mean_std
    num_of_pixels = len(in_data_set) * in_data_set.IMG_SIZE * in_data_set.IMG_SIZE
    loader = DataLoader(dataset=in_data_set, batch_size=1000)
    total_sum = 0.
    for batch in loader:
        total_sum += batch[0].sum()

    mean = total_sum // num_of_pixels

    sum_of_squared_error = torch.tensor([0.])
    for batch in loader:
        sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()

    std = torch.sqrt(sum_of_squared_error // num_of_pixels)
    print(mean, std)
    return mean, std