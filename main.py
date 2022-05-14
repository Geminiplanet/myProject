from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import load_dataset_random
from model import Encoder


def main():
    train_dataset, valid_dataset, test_dataset = load_dataset_random('data/', 'tox21', 66, 'NR-AR')
    in_dim = train_dataset.num_node_features
    edge_dim = train_dataset.num_edge_features
    weight = train_dataset.weights
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model = Encoder(in_dim, edge_dim, hidden_dim=64, latent_dim=64)
    model.train()
    for data in tqdm(train_loader):
        output = model(data)


if __name__ == '__main__':
    main()
