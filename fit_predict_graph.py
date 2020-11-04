from torch_geometric.nn import GCNConv, JumpingKnowledge, global_add_pool
from torch.nn import functional as F
from torch_geometric import transforms
from skorch import NeuralNetClassifier
import torch
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import InMemoryDataset, download_url
from rdkit import Chem
import pandas as pd
import torch


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

    X, y = torch.arange(len(covid)).long(), covid.data.y
    net = NeuralNetClassifier(
        module=SimpleGNN,
        module__dataset=covid,
        max_epochs=20,
        batch_size=-1,
        lr=0.001
    )
    print("Starting training")
    fit = net.fit(X_train, y_train)
    from sklearn.metrics import roc_auc_score
    print('AUC TRAIN', roc_auc_score(
        y_train, fit.predict_proba(X_train)[:, 0]))
    print('AUC TEST', roc_auc_score(y_test, fit.predict_proba(X_test)[:, 0]))
    print('MAP TEST', average_precision_score(y_test, fit.predict_proba(X_test)[:, 0]))
