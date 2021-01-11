'''
OSM amenity에 있는 POI들을 원핫인코딩 하기
'''
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os 


main_path = os.getcwd()

excel_file = os.path.join(main_path, 'amenity.xls')
amenity_df = pd.read_excel(excel_file)
amenity_df['value'][amenity_df['value'] == 'café'] = 'cafe'

# Encoding categorical datas using Label Encoder
# 순서의 의미가 있을 때, 고유값 개수가 많을 때
le = LabelEncoder()
amenity_df['label'] = le.fit_transform(amenity_df['value'])


# Encoding categorical data using One Hot Encoder
# 순서가 없을 때, 고유값 개수가 많지 X
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])],
                       remainder='passthrough')
amenity_list = amenity_df['value'].to_frame()
amenity_oh = ct.fit_transform(amenity_list).toarray()