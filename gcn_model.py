# gcn_model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gcn import add_features_, dataloader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import ClusterData, ClusterLoader

data_, G = dataloader()
data_ = add_features_(data_, G)
dataset = data_
# dataset = InMemoryDataset.collate(data)
cluster_data = ClusterData(data_, num_parts=50, recursive=False)
test_mask = cluster_data
train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True,
							 num_workers=12)


class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = GCNConv(dataset.num_node_features, 16)
		# self.conv2 = GCNConv(16, dataset.num_classes)
		self.conv2 = GCNConv(16, 1)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)

		return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data_
test_mask = data_
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()

for batch in train_loader:
	batch = batch.to(device)
	optimizer.zero_grad()
	out = model(batch.x, batch.edge_index)
	loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
	loss.backward()
	optimizer.step()


model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[test_mask].eq(data.y[test_mask]).sum().item())
acc = correct / int(test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))