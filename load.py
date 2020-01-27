import pandas as pd
import streamlit as st
import geopandas as gpd


@st.cache
def load_data():
    data = pd.read_csv('./datasets/wine-reviews/winemag-data_first150k.csv')
    data.drop('Unnamed: 0', inplace=True, axis=1)
    print(data)
    return data


@st.cache
def load_shape_map():
    data = gpd.read_file('./datasets/worldmapshapes/ne_110m_admin_0_countries.shp')[['ADMIN', 'geometry']]
    data.columns = ['country', 'geometry']
    return data
