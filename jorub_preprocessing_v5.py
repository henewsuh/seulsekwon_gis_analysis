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
from jorub_categorical_encoding import amenity_df



geolocoder = Nominatim(user_agent = 'South Korea')


def geocoding(address): 
    geo = geolocoder.geocode(address)
    crd = (geo.latitude, geo.longitude)
    print(crd)
    return crd




# osmnx.pois.pois_from_address로 해당 좌표 인근의 POI 정보 불러오기 
tags = {'amenity': True}

address_list = ['서울대입구역',
'도림로 264',
'현충로 213']


crds = []
demo = dict()
crd = geocoding(address_list[0]) # address_list 리스트의 주소 중 index 0의 주소 경위도 좌표 받아오기
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
        
        
G = ox.graph_from_point(crds[0], dist=500, network_type="walk")        
house = ox.distance.get_nearest_node(G, crds[0], method='euclidean')
house_crd = (crds[0][1], crds[0][0])
house_dist = ox.distance.euclidean_dist_vec(crds[0][0], crds[0][1], G.nodes[house]['y'], G.nodes[house]['x'])        


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
G_j = nx.Graph()   
G_j.add_nodes_from([(house, {'label' : 999, 'dist' : 0, 'type': 'house'})])    
amenity_le = amenity_df


for i in range(len(nearest_to_pois)): 
    near_dist = ox.distance.euclidean_dist_vec(nearest_to_pois.iloc[i]['poi_x'], nearest_to_pois.iloc[i]['poi_y'], 
                                               nearest_to_pois.iloc[i]['x'], nearest_to_pois.iloc[i]['y'])        

    orig_node = house 
    target_node = nearest_to_pois.iloc[i]['osmid']
    path_dist = nx.shortest_path_length(G, source=orig_node, target=target_node, weight='length')
    
    total_dist = near_dist + path_dist + house_dist
       
   
    poi_osmid = nearest_to_pois.iloc[i]['poi_osmid']
    amenity_value = pois['amenity'][pois['osmid'] == poi_osmid].values[0]
    le =  amenity_le['label'][amenity_le['value'] == amenity_value].values[0]

    
    G_j.add_nodes_from([(poi_osmid, {'label' : le, 'dist' : total_dist, 'type': amenity_value})])
    G_j.add_weighted_edges_from([(house, poi_osmid, total_dist)])
   

   
pos = nx.spring_layout(G_j) # pos = nx.nx_agraph.graphviz_layout(G)
nx.draw_networkx(G_j, pos, font_size=5, node_size=10)
labels = nx.get_edge_attributes(G_j, 'weight')
nx.draw_networkx_edge_labels(G_j, pos, font_size=5, edge_labels=labels)
    
















    
       