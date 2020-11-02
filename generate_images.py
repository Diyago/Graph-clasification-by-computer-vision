from torch_geometric.nn import GCNConv, JumpingKnowledge, global_add_pool
from torch.nn import functional as F
from torch_geometric import transforms
from skorch import NeuralNetClassifier
import torch
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch_geometric.data import InMemoryDataset, download_url
from rdkit import Chem
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric import utils
import matplotlib.pyplot as plt
import networkx as nx


class COVID(InMemoryDataset):
    url = 'https://github.com/yangkevin2/coronavirus_data/raw/master/data/mpro_xchem.csv'

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(COVID, self).__init__(root, transform, pre_transform, pre_filter)
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['mpro_xchem.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        data_list = []
        for smiles, label in df.itertuples(False, None):
            mol = Chem.MolFromSmiles(smiles)  # Read the molecule info
            adj = Chem.GetAdjacencyMatrix(mol)  # Get molecule structure
            # You should extract other features here!
            data = Data(num_nodes=adj.shape[0],
                        edge_index=torch.Tensor(adj).nonzero().T,  y=label)
            data_list.append(data)
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


class SimpleGNN(torch.nn.Module):
    def __init__(self, dataset, hidden=64, layers=6):
        super(SimpleGNN, self).__init__()
        self.dataset = dataset
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels=dataset.num_node_features,
                                  out_channels=hidden))

        for _ in range(1, layers):
            self.convs.append(GCNConv(in_channels=hidden, out_channels=hidden))

        self.jk = JumpingKnowledge(mode="cat")
        self.jk_lin = torch.nn.Linear(
            in_features=hidden*layers, out_features=hidden)
        self.lin_1 = torch.nn.Linear(in_features=hidden, out_features=hidden)
        self.lin_2 = torch.nn.Linear(
            in_features=hidden, out_features=dataset.num_classes)

    def forward(self, index):
        data = Batch.from_data_list(self.dataset[index])
        x = data.x
        xs = []
        for conv in self.convs:
            x = F.relu(conv(x=x, edge_index=data.edge_index))
            xs.append(x)

        x = self.jk(xs)
        x = F.relu(self.jk_lin(x))
        x = global_add_pool(x, batch=data.batch)
        x = F.relu(self.lin_1(x))
        x = F.softmax(self.lin_2(x), dim=-1)
        return x


if __name__ == "__main__":
    print("Preprocessing data")
    ohd = transforms.OneHotDegree(max_degree=4)
    covid = COVID(root='./data/COVID/', transform=ohd)

    X_train, X_test, y_train, y_test = train_test_split(
        torch.arange(len(covid)).long(), covid.data.y, test_size=0.3, random_state=42)

    print("Generating for train images")
    for graph in tqdm(X_train):
        fig = plt.figure(figsize=(6, 6))
        G = utils.to_networkx(covid[int(graph)])
        a = nx.draw_kamada_kawai(G)
        plt.savefig("./train/id_{}_y_{}.jpg".format(int(graph),
                                                    covid.data.y[int(graph)]), format="jpg")

    print("Generating for test images")
    for graph in tqdm(X_test):
        fig = plt.figure(figsize=(6, 6))
        G = utils.to_networkx(covid[int(graph)])
        a = nx.draw_kamada_kawai(G)
        plt.savefig("./test/id_{}_y_{}.jpg".format(int(graph),
                                                   covid.data.y[int(graph)]), format="jpg")
