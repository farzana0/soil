# gcn_model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GENConv, global_mean_pool
from gcn import add_features_, dataloader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import ClusterData, ClusterLoader
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.transforms import Distance
from torch_geometric.data import DataLoader


num_epochs = 10
data_, G = dataloader()
data_ = add_features_(data_, G)
tr = Distance()
data_ = tr(data_)
dataset = data_
print(dataset)
# dataset = InMemoryDataset.collate(data)
cluster_data = ClusterData(data_, num_parts=12, recursive=False)
test_mask = cluster_data
train_loader = ClusterLoader(cluster_data, batch_size=3, shuffle=True,
							 num_workers=12)
# train_loader = DataLoader(data_, batch_size=1, shuffle=True)

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = GENConv(dataset.num_node_features, 32, aggr = "power")
		# self.conv2 = GCNConv(16, dataset.num_classes)
		self.conv3 = GENConv(32, 16, aggr = "power")
		self.conv2 = GENConv(16, dataset.y.shape[1], aggr = "power")

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv3(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)

		return F.relu(x)
		# return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data_.to(device)
test_mask = data_
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

def train():
	model.train()
	loss_all=0
	for batch in train_loader:
		batch = batch.to(device)
		optimizer.zero_grad()
		out = model(batch)
		loss = F.mse_loss(out, batch.y)
		loss.backward()
		loss_all +=  loss.item()
		optimizer.step()
		print(loss_all)


for epoch in range(num_epochs):
	train()



model.eval()
pred = model(data)
# print(data.y.shape)
# print(pred.shape)
mse = mean_squared_error(data.y.detach().numpy(), pred.detach().numpy(), multioutput='uniform_average')
r_square = r2_score(data.y.detach().numpy(), pred.detach().numpy(), multioutput= 'uniform_average')
print(mse)
print(r_square)
