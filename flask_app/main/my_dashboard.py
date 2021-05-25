import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.offline as py
from . import covid_info


#dir_path = 'D:\\My_study_for_data_science\\python\\covid_final\\covid_final\\covid_final\\flask_app\\templates\\'


#Change dataset and change static info to variable name
def get_choropleth_dia():
    dataset_url = 'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
    df = pd.read_csv(dataset_url)

    fig = px.choropleth(df , locations = 'Country', locationmode = 'country names', color = 'Confirmed'
                        ,animation_frame = 'Date')
    fig.update_layout(title_text = 'Global spread of Covid19')
    file_name = covid_info.get_dir_path('Global_covid.html')
    py.offline.plot(fig, filename=file_name, auto_open=False)
    return 'Global_covid.html'