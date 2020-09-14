from scipy import spatial
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import pickle

df = pd.read_csv('Site_Inv1990.csv', usecols = ['xcoord', 'ycoord'], dtype=np.float64)
id_xy = pd.read_csv('Site_Inv1990.csv', usecols = ['IDEN2'], dtype=str)['IDEN2'].to_numpy()
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.xcoord, y=df.ycoord)).to_crs('+proj=robin')
# df[['xcoord', 'ycoord']] = df[['xcoord', 'ycoord']].convert_objects(convert_numeric=True)
dfx = df['xcoord'].to_numpy()
dfy = df['ycoord'].to_numpy()
# id_xy = df['ycoord'].to_numpy()
# centroids = df.to_numpy()
centroids = np.column_stack([dfx, dfy])

# We define the range
radius=100
# Like in the previous example we populate the KD-tree
kdtree = spatial.cKDTree(centroids)
neigh_list = {}
edge_list =[]

# We cycle on every point and calculate its neighbours 
# with the function query_ball_point
spatial_graph = nx.Graph()

for m, g in enumerate(centroids):
	neigh_list[m] = (kdtree.query_ball_point(g, r=radius))
	for item in neigh_list[m][:-1]:
		spatial_graph.add_edge(id_xy[m], id_xy[item])

# Dump graph
with open("spatial_graph.pickle", 'wb') as f:
    pickle.dump(spatial_graph, f)

# f.close()


# print(neigh_list)