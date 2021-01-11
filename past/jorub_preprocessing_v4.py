import networkx as nx 
import torch 
import dgl 
import matplotlib.pyplot as plt 
from geopy.geocoders import Nominatim
import osmnx as ox 
import shapely 
import pandas as pd
import pandana as pdna
from shapely import geometry
from descartes.patch import PolygonPatch
from shapely.geometry import LineString



# http://kuanbutts.com/2017/08/08/how-to-pdna/


geolocoder = Nominatim(user_agent = 'South Korea')


def geocoding(address): 
    geo = geolocoder.geocode(address)
    crd = (geo.latitude, geo.longitude)
    print(crd)
    return crd



def multiG_to_G(MG, aggr_func=max):
    rG = nx.Graph()
    # add nodes
    rG.add_nodes_from([n for n in MG.nodes()])
    # extract edge and their weights
    edge_weight_dict = {}
    for u, v, e_attr in MG.edges(data=True):
        e, weight = (u, v), e_attr['weight']
        if e in edge_weight_dict:
            edge_weight_dict[e].append(weight)
        else:
            edge_weight_dict[e] = [weight]
    # add edges by aggregating their weighy by `aggr_func`
    for e, e_weight_lst in edge_weight_dict.items():
        rG.add_edge(*e, weight=aggr_func(e_weight_lst))
    return rG
    

# osmnx.pois.pois_from_address로 해당 좌표 인근의 POI 정보 불러오기 
tags = {'amenity': True}

address_list = ['서울대입구역',
'도림로 264',
'현충로 213']


crds = []
demo = dict()
crd = geocoding(address_list[1]) # address_list 리스트의 주소 중 index 0의 주소 경위도 좌표 받아오기
crds.append(crd)
pois = ox.pois.pois_from_point(crd, tags=tags, dist=500) # 해당 경위도 좌표계 인근 500m의 POI들을 받아오기 
pois = pois.dropna(axis=0, subset=['name'])
pois.reset_index(inplace=True)


pois_df = pd.DataFrame(index=range(0, len(pois)), columns=['poi_osmid', 'poi_x', 'poi_y'])
amenity_coord = dict()
for i in range(len(pois)):
    shapely_obj = pois.loc[i, ['geometry']]
    
    if shapely_obj[0].type == 'Point':
        x_crd = shapely_obj[0].xy[0][0]
        y_crd = shapely_obj[0].xy[1][0]
        xy_crd = (x_crd, y_crd)
        amenity_coord[pois.loc[i, ['name']][0]] = [pois.loc[i, ['amenity']][0], xy_crd]
        pois_df.loc[i] = [pois.loc[i, ['osmid']][0], x_crd, y_crd]
    
    if shapely_obj[0].type == 'Polygon': 
        x_crd = shapely_obj[0].centroid.xy[0][0]
        y_crd = shapely_obj[0].centroid.xy[1][0]
        xy_crd = (x_crd, y_crd)
        amenity_coord[pois.loc[i, ['name']][0]] =  [pois.loc[i, ['amenity']][0], xy_crd]
        pois_df.loc[i] = [pois.loc[i, ['osmid']][0], x_crd, y_crd]


poi_count = pois['amenity'].value_counts() # pois_df에 저장된 각 POI amenity 종류 별로 몇개씩 있는지 카운트    
# demo[address_list[1]] = poi_count # 아직은 사용하고 있지 않음. 나중에는 각 주소들을 key로, poi_count를 value로 갖는 dictionary 생성할 것임

G = ox.graph_from_point(crds[0], dist=500, network_type="walk") 
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)


G_original = G.to_undirected().copy()
house = ox.distance.get_nearest_node(G_original, crds[0], method='euclidean')
G.add_node(house)
fig, ax = ox.plot_graph(G, show=False, close=False,
                        edge_color='#777777')
ax.set_facecolor('white')
# ox.plot_graph(G_projected)


edge_list = list(G.edges) #G의 엣지들을 리스트로 저장
node_list = list(G.nodes) #G의 노드들을 리스트로 저장
print(' ')
print('Original Nodes # : {}'.format(len(node_list)))
print('Original Edges # : {}'.format(len(edge_list)))
print(' ')


node_dict = dict() # osmid를 key로, 해당 노드의 (x, y) 좌표를 튜플로 하는 dictionary 생성
for i in range(len(node_list)):
    osmid = node_list[i] 
    node_dict[osmid] = (str(osmid), G.nodes[osmid]['x'], G.nodes[osmid]['y'])

edge_dict = dict() # key는 인덱스, value에 딕셔너리 -- {시작노드:(x,y), 끝노드:(x,y)} 
for i in range(len(edge_list)):
    st_osmid = edge_list[i][0]
    ed_osmid = edge_list[i][1]
    st_crd = (G.nodes[st_osmid]['x'], G.nodes[st_osmid]['y'])
    ed_crd = (G.nodes[ed_osmid]['x'], G.nodes[ed_osmid]['y'])
    crd_dict = dict()
    crd_dict[st_osmid] = st_crd
    crd_dict[ed_osmid] = ed_crd
    edge_dict[i] = crd_dict


# 위에서 생성한 node_dict랑 edge_dict로 데이터프레임 생성 
node_df = pd.DataFrame(node_dict).T
node_df.columns = ['osmid','x', 'y']
node_df['osmid'] = node_df['osmid'].astype('int64')

edge_df = pd.DataFrame(index=range(0, len(edge_dict)), columns=['st_osmid', 'st_x', 'st_y', 'ed_osmid', 'ed_x', 'ed_y', 'edge_weight'])
for i in range(len(edge_dict)):
    k, v = edge_dict[i].items()
    st_osmid = k[0]
    ed_osmid = v[0]
    st_x = k[1][0]
    st_y = k[1][1]
    ed_x = v[1][0]
    ed_y = v[1][1]
    edge_weight = 1
    edge_df.loc[i] = [st_osmid, st_x, st_y, ed_osmid, ed_x, ed_y, edge_weight]
    

net = pdna.Network(node_df['x'], node_df['y'], edge_df['st_osmid'], edge_df['ed_osmid'], edge_df[['edge_weight']])

near_ids = net.get_node_ids(pois_df['poi_x'],
                            pois_df['poi_y'],
                            mapping_distance=1)
pois_df['nearest_node_id'] = near_ids
nearest_to_pois = pd.merge(pois_df,
                           node_df,
                           left_on='nearest_node_id',
                           right_on='osmid',
                           how='left',
                           sort=False,
                           suffixes=['_from', '_to'])  
    
col = list(nearest_to_pois.columns)
for i in range(len(nearest_to_pois)): 
    y1 = nearest_to_pois.iloc[i]['poi_y']
    x1 = nearest_to_pois.iloc[i]['poi_x']
    y2 = nearest_to_pois.iloc[i]['y']
    x2 = nearest_to_pois.iloc[i]['x']
    distance = ox.distance.euclidean_dist_vec(y1, x1, y2, x2)
    print(distance)
    line = LineString([(x1, y1), (x2, y2)])
    new_data = {'osmid': '{}'.format(i),
                'name': 'new',
                'highway': 'nan',
                'oneway': 'nan',
                'length': distance, 
                'geometry': line,
                'service': 'nan',
                'tunnel': 'nan',
                'lanes': 'nan',
                'u': nearest_to_pois.iloc[i]['poi_osmid'],
                'v': nearest_to_pois.iloc[i]['osmid'],
                'key': 0
        }
    edges = edges.append(new_data, ignore_index=True) 

    
# add poi (node) to node_df 
pois_df_ = dict()
for i in range(len(pois_df)):
    poi_osmid = pois_df.loc[i, ['poi_osmid']][0]
    poi_x = pois_df.loc[i, ['poi_x']][0]
    poi_y = pois_df.loc[i, ['poi_x']][0]
    pois_df_[i] = [poi_osmid, poi_x, poi_y]
pois_df_ = pd.DataFrame(pois_df_).T
pois_df_.columns = ['osmid','x', 'y']
pois_df_['osmid'] = pois_df_['osmid'].astype('int64')
nnode_df = pd.concat([node_df, pois_df_]) # nnode_df는 기존 node_df에 poi_df를 추가한 데이터프레임
nnode_df = nnode_df.reset_index()
nnode_df = nnode_df.drop(['index'], axis=1)



poi_edge_list = []
addi_edge_df = dict() # poi와 인근노드를 연결하면서 추가된 엣지 df
for i in range(len(pois_df)):
    poi_n = pois_df.loc[i, ['poi_osmid']][0] #st_osmid
    nearest_n = pois_df.loc[i, ['nearest_node_id']][0] #ed_osmid
    st_x = nnode_df.loc[nnode_df['osmid'] == poi_n]['x'].item()
    st_y = nnode_df.loc[nnode_df['osmid'] == poi_n]['y'].item()
    ed_x = nnode_df.loc[nnode_df['osmid'] == nearest_n]['x'].item()
    ed_y = nnode_df.loc[nnode_df['osmid'] == nearest_n]['y'].item()
    edge_weight = 1
    addi_edge_df[i] = [poi_n, st_x, st_y, nearest_n, ed_x, ed_y, edge_weight]
    poi_edge_list.append((poi_n, nearest_n, 0))
addi_edge_df = pd.DataFrame(addi_edge_df).T
addi_edge_df.columns = ['st_osmid','st_x', 'st_y', 'ed_osmid', 'ed_x', 'ed_y', 'edge_weight']
eedge_df = pd.concat([edge_df, addi_edge_df])



# add_edge_df의 새로 생성된 엣지들을 기존의 G에 추가
# poi들을 기존의 G에 추가 
print('POIs (node, edge) from OSM added!')
poi_node_list = list(pois_df['poi_osmid'])
G.add_nodes_from(poi_node_list)
G.add_edges_from(poi_edge_list)
G_undirected = G.to_undirected()


print(' ')
print('New Nodes # : {}'.format(len(list(G.nodes))))
print('New Edges # : {}'.format(len(list(G.edges))))
print(' ')


# making real graph
from jorub_categorical_encoding import amenity_df
amenity_le = amenity_df
j_node_ls = poi_node_list.copy()
j_node_ls.append(house)


route = nx.shortest_path(G_undirected, source=house, target=368654020, weight='length')

fig, ax = ox.plot.plot_graph_route(G_undirected, route=route, route_color='r')


G_j = nx.Graph()
for i in range(len(j_node_ls)):
    
    if j_node_ls[i] == house:
        continue
    else: 
        osmid = j_node_ls[i]
        amenity_value = pois['amenity'][pois['osmid'] == osmid].values[0]
        le =  amenity_le['label'][amenity_le['value'] == amenity_value].values[0]
        G_j.add_nodes_from([(osmid, {'label' : le})])
        
        distance = nx.shortest_path_length(G_undirected, house, osmid, weight='length')
        print(distance)
        













