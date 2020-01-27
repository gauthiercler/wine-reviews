import streamlit as st
from mpl_toolkits.axes_grid1 import make_axes_locatable

from load import load_data, load_shape_map
import seaborn as sns
import matplotlib.pyplot as plt


def eda(data):
    st.title('Exploratory data analysis')
    st.dataframe(data.head(100))

    with st.echo():
        price_distrib = sns.distplot(data[data['price'] < 300]['price'])
    price_distrib.set_xlabel("Price", fontsize=15)
    price_distrib.set_ylabel("Frequency", fontsize=15)
    st.pyplot()

    with st.echo():
        count_plot = sns.countplot(x='points', data=data)
    count_plot.set_xlabel('Points', fontsize=15)
    count_plot.set_ylabel('Count', fontsize=15)
    st.pyplot()

    with st.echo():
        price_group = data.groupby(['points']).mean()
        plt.plot(price_group.index, price_group.values, c='red')
    plt.xlabel('Points', fontsize=15)
    plt.ylabel('Price', fontsize=15)
    st.pyplot()

    with st.echo():
        geoshape = load_shape_map()
        grouped = data.groupby(by=['country'], as_index=False).count()
        grouped.country.replace({
            'England': 'United Kingdom',
            'US': 'United States of America',
            'Serbia': 'Republic of Serbia',
            'Czech Republic': 'Czechia'
        }, inplace=True)
        merged = geoshape.merge(grouped, on='country', how='left')
        merged.plot(column='points', cmap='Blues', linewidth=0.8, edgecolor='0.8', legend=True,
                    legend_kwds={'label': "Number of wines by Country"})
    st.pyplot()

    with st.echo():
        na_sum = data.isna().sum()
    st.write(na_sum)

