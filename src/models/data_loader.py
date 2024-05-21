import numpy as np
import torch

def load_data(data_path):
    x_data = np.load(f'{data_path}')
    VE_data = np.load(f'{data_path}')
    return x_data, VE_data

def get_dataloader(data_path, batch_size):
    x_data, VE_data = load_data(data_path)

    # Convert data to PyTorch tensors
    x_data = torch.tensor(x_data, dtype=torch.float32).reshape(100, 3, 396)
    VE_data = torch.tensor(VE_data, dtype=torch.float32).reshape(100, 3, 396)

    print("x_data shape:", x_data.shape)
    print("VE_data shape:", VE_data.shape)

    # Prepare the dataset and DataLoader
    dataset = torch.utils.data.TensorDataset(x_data, VE_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
