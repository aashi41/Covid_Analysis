from covid import Covid
import os,sys
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
plt.style.use('fivethirtyeight')
#%matplotlib inline

covid = Covid()
covid1 = Covid(source="worldometers")


def get_world_total_count():
    active = covid.get_total_active_cases()
    confirmed = covid.get_total_confirmed_cases()
    recovered = covid.get_total_recovered()
    deaths = covid.get_total_deaths()
    total_cnts = {'ACTIVE' : active, 'CONFIRMED': confirmed, 'DEATHS': deaths, 'RECOVERED': recovered}
    return total_cnts

def get_country_list():
    #covid_info = covid.get_data()
    '''countries_id = covid.list_countries()
    countries = list()
    for i in countries_id:
        countries.append(i['name'])
    return countries  '''
    #cntry_list = covid1.list_countries()#for dropdown
    #final_cntry_list = [i.capitalize() for i in cntry_list if i.strip() != '']
    final_cntry_list=confirmed_df['Country/Region'].unique().tolist()
    return final_cntry_list




def get_countrywise_total_count(country='India'):
    country_status = covid.get_status_by_country_name(country)
    rem_list = ['id','latitude','longitude','last_update']
    [country_status.pop(key, None) for key in rem_list]
    return country_status
    

#dir_path = 'D:\\My_study_for_data_science\\python\\covid_final\\covid_final\\covid_final\\flask_app\\static\\'
def get_dir_path(path):
    base_path = os.path.dirname(__file__)
    dir_path = os.path.abspath(os.path.join(base_path,"..","static",path))
    return dir_path


daily_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/05-15-2021.csv')
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = death_df.loc[:, cols[4]:cols[-1]]
recovered = recovered_df.loc[:, cols[4]:cols[-1]]

confirmed_df_grpby = confirmed_df
confirmed_df_grpby = confirmed_df_grpby.groupby('Country/Region').sum()
confirmed_df_grpby = confirmed_df_grpby.reset_index()


death_df_grpby = death_df
death_df_grpby = death_df_grpby.groupby('Country/Region').sum()
death_df_grpby = death_df_grpby.reset_index()

recovered_df_grpby = recovered_df
recovered_df_grpby = recovered_df_grpby.groupby('Country/Region').sum()
recovered_df_grpby = recovered_df_grpby.reset_index()



#df = confirmed_df_grpby.drop(confirmed_df_grpby.iloc[:, 1:-1], axis = 1) #excluding -1
#df = df.sort_values(cols[-1], ascending=False).head()

def draw_pie_diagram(data,title,path):
    fig = px.pie(data, values=cols[-1], names=cols[1], title=title)
    pio.write_html(fig, file=path, auto_open=False,full_html=False)
    #fig.show()
    

def draw_bar_diagram(data,title,path):
    fig1 = px.bar(data, y=cols[-1], x=cols[1], title=title,color=cols[-1])
    pio.write_html(fig1, file=path, auto_open=False,full_html=False)

def getDiagram_top5_corona_affected_cnty_comparison(dia='pie'):
    World_total_recent_confirmed = confirmed_df[cols[-1]].sum()
    World_total_recent_death = death_df[cols[-1]].sum()
    World_total_recent_recovered = recovered_df[cols[-1]].sum()

    top_five_recent_confirmed = confirmed_df_grpby.sort_values(cols[-1],ascending=False).head()[[cols[1],cols[-1]]]
    other_cnt = int(World_total_recent_confirmed) - int(top_five_recent_confirmed.sum().to_list()[1])
    top_five_recent_confirmed.loc[len(top_five_recent_confirmed.index)] = ['Others',other_cnt] 

    top_five_recent_deaths = death_df_grpby.sort_values(cols[-1],ascending=False).head()[[cols[1],cols[-1]]]
    other_cnt_death = int(World_total_recent_death) - int(top_five_recent_deaths.sum().to_list()[1])
    top_five_recent_deaths.loc[len(top_five_recent_deaths.index)] = ['Others',other_cnt_death] 

    top_five_recent_recovered = recovered_df_grpby.sort_values(cols[-1],ascending=False).head()[[cols[1],cols[-1]]]
    other_cnt_recovered = int(World_total_recent_recovered) - int(top_five_recent_recovered.sum().to_list()[1])
    top_five_recent_recovered.loc[len(top_five_recent_recovered.index)] = ['Others',other_cnt_recovered]   
  
    dia_names = []
    if(dia == 'pie'):
        draw_pie_diagram(top_five_recent_confirmed,'Confirmed Corona cases worldwide',get_dir_path('top_five_recent_confirmed_pie.html'))
        draw_pie_diagram(top_five_recent_deaths,'Deaths due to Corona cases worldwide',get_dir_path('top_five_recent_deaths_pie.html'))
        draw_pie_diagram(top_five_recent_recovered,'Recovered Corona cases worldwide',get_dir_path('top_five_recent_recovered_pie.html'))
        dia_names=['top_five_recent_confirmed_pie.html','top_five_recent_deaths_pie.html','top_five_recent_recovered_pie.html']
    elif(dia == 'bar'):
        draw_bar_diagram(top_five_recent_confirmed,'Corona cases comparison among top 5 countries',get_dir_path('top_five_recent_confirmed_bar.html'))
        draw_bar_diagram(top_five_recent_deaths,'Corona cases death comparison among top 5 countries',get_dir_path('top_five_recent_deaths_bar.html'))
        draw_bar_diagram(top_five_recent_recovered,'Corona cases recovered comparison among top 5 countries',get_dir_path('top_five_recent_recovered_bar.html'))
        dia_names=['top_five_recent_confirmed_bar.html','top_five_recent_deaths_bar.html','top_five_recent_recovered_bar.html']
    return dia_names


#getDiagram_top5_corona_affected_cnty_comparison()


# Creating pie plot
'''def draw_pie_diagram(data,labels,cols,title,path):
    print('Inside draw_pie')
    fig = plt.figure(figsize =(10, 7))
    my_explode = (0.1, 0, 0, 0, 0 ,0)
    plt.pie(data, labels = labels,autopct='%1.1f%%', startangle=50, shadow = True,explode=my_explode)
    plt.title(title+ 'Date (mm/dd/yy): '+ cols[-1])
    plt.axis('equal')
    #plt.show()
    print(path)
    plt.savefig(path)

# creating the bar plot
def draw_bar_diagram(x,y,title,path):
    fig = plt.figure(figsize = (12, 8))    
    plt.bar(x,y, color ='orange',width = 0.4)   
    plt.xlabel("Countries")
    plt.ylabel("Corona cases")
    plt.title(title+ 'Date (mm/dd/yy): '+ cols[-1])
    print(path)
    plt.savefig(path)

## Get top 5 affected countries comparison    
def getDiagram_top5_corona_affected_cnty_comparison(dia='pie'):
    print('inside dia')
    World_total_recent_confirmed = confirmed_df[cols[-1]].sum()
    World_total_recent_death = death_df[cols[-1]].sum()
    World_total_recent_recovered = recovered_df[cols[-1]].sum()

    top_five_recent_confirmed = confirmed_df.sort_values(cols[-1],ascending=False).head()[[cols[1],cols[-1]]]
    top_five_recent_confirmed
    other_cnt = int(World_total_recent_confirmed) - int(top_five_recent_confirmed.sum().to_list()[1])
    top_five_recent_confirmed.loc[len(top_five_recent_confirmed.index)] = ['Others',other_cnt] 

    top_five_recent_deaths = death_df.sort_values(cols[-1],ascending=False).head()[[cols[1],cols[-1]]]
    other_cnt_death = int(World_total_recent_death) - int(top_five_recent_deaths.sum().to_list()[1])
    top_five_recent_deaths.loc[len(top_five_recent_deaths.index)] = ['Others',other_cnt_death] 

    top_five_recent_recovered = recovered_df.sort_values(cols[-1],ascending=False).head()[[cols[1],cols[-1]]]
    other_cnt_recovered = int(World_total_recent_recovered) - int(top_five_recent_recovered.sum().to_list()[1])
    top_five_recent_recovered.loc[len(top_five_recent_recovered.index)] = ['Others',other_cnt_recovered]   
  
    dia_names = []
    if(dia == 'pie'):
        draw_pie_diagram(top_five_recent_confirmed[cols[-1]],top_five_recent_confirmed[cols[1]],cols,'Confirmed Corona cases worldwide',dir_path+'top_five_recent_confirmed_pie.png')
        draw_pie_diagram(top_five_recent_deaths[cols[-1]],top_five_recent_deaths[cols[1]],cols,'Deaths due to Corona cases worldwide',dir_path+'top_five_recent_deaths_pie.png')
        draw_pie_diagram(top_five_recent_recovered[cols[-1]],top_five_recent_recovered[cols[1]],cols,'Recovered Corona cases worldwide',dir_path+'top_five_recent_recovered_pie.png')
        dia_names=['top_five_recent_confirmed_pie.png','top_five_recent_deaths_pie.png','top_five_recent_recovered_pie.png']
    elif(dia == 'bar'):
        draw_bar_diagram(top_five_recent_confirmed[cols[1]].iloc[:5], top_five_recent_confirmed[cols[-1]].iloc[:5],'Corona cases comparison among top 5 countries',dir_path+'top_five_recent_confirmed_bar.png')
        draw_bar_diagram(top_five_recent_deaths[cols[1]].iloc[:5], top_five_recent_deaths[cols[-1]].iloc[:5],'Corona cases death comparison among top 5 countries',dir_path+'top_five_recent_deaths_bar.png')
        draw_bar_diagram(top_five_recent_recovered[cols[1]].iloc[:5], top_five_recent_recovered[cols[-1]].iloc[:5],'Corona cases recovered comparison among top 5 countries',dir_path+'top_five_recent_recovered_bar.png')
        dia_names=['top_five_recent_confirmed_bar.png','top_five_recent_deaths_bar.png','top_five_recent_recovered_bar.png']
    return dia_names'''

'''def draw_line_plot(dates,data1,data2,data3,country,path):
    plt.figure(figsize= (15,10))
    plt.xticks(rotation = 90 ,fontsize = 11)
    plt.yticks(fontsize = 10)
    plt.xlabel("Dates",fontsize = 20)
    plt.ylabel('Total cases',fontsize = 20)
    plt.title("Total Confirmed, Active, Death in "+ country , fontsize = 20)

    ax1 = plt.plot_date(y= data1,x= dates,label = 'Confirmed',linestyle ='-',color = 'b')
    ax2 = plt.plot_date(y= data2,x= dates,label = 'Recovered',linestyle ='-',color = 'g')
    ax3 = plt.plot_date(y= data3,x= dates,label = 'Death',linestyle ='-',color = 'r')
    plt.legend()
    plt.savefig(path)




def getDiagram_Country_analysis(country):
    india_confirmed=confirmed_df_grpby.loc[confirmed_df_grpby['Country/Region']==country].iloc[:,4:]
    india_deaths=death_df_grpby.loc[death_df_grpby['Country/Region']==country].iloc[:,4:]
    india_recovered=recovered_df_grpby.loc[recovered_df_grpby['Country/Region']==country].iloc[:,4:]

    dates = list(confirmed_df_grpby.columns[4:])
    dates = list(pd.to_datetime(dates))
    dia_name = country+'_analysis.png'
    draw_line_plot(dates,india_confirmed.iloc[0],india_recovered.iloc[0],india_deaths.iloc[0],country,dir_path+dia_name)
    return dia_name
    '''

def draw_line_plot(dates,data1,data2,data3,country,path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=data1, name='Confirmed cases',
                             mode='lines+markers',line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=dates, y=data2, name='Death cases',
                             mode='lines+markers',line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=dates, y=data3, name='Recovered cases',
                             mode='lines+markers',line=dict(color='green', width=4)))
    
    fig.update_layout(title="Total Confirmed, Active, Death in "+ country,
                   xaxis_title='dates',
                   yaxis_title='Total cases')
    print('Country wise diagram')
    pio.write_html(fig, file=path, auto_open=False,full_html=False)


    
def getDiagram_Country_analysis(country):
    india_confirmed=confirmed_df_grpby.loc[confirmed_df_grpby['Country/Region']==country].iloc[:,4:]
    india_deaths=death_df_grpby.loc[death_df_grpby['Country/Region']==country].iloc[:,4:]
    india_recovered=recovered_df_grpby.loc[recovered_df_grpby['Country/Region']==country].iloc[:,4:]

    dates = list(confirmed_df_grpby.columns[4:])
    dates = list(pd.to_datetime(dates))
    dia_name = country+'_analysis.html'
    path = get_dir_path(dia_name)
    draw_line_plot(dates,india_confirmed.iloc[0],india_deaths.iloc[0],india_recovered.iloc[0],country,path)
    return dia_name
    




def getRateAnalysis(Country):# we can get old data as well
    confirmed_country = confirmed_df_grpby.loc[confirmed_df_grpby['Country/Region']=='India']
    confirmed_sum = confirmed_country[cols[-1]]

    death_country = death_df_grpby.loc[death_df_grpby['Country/Region']=='India']
    death_sum = death_country[cols[-1]]

    recovered_country = recovered_df_grpby.loc[recovered_df_grpby['Country/Region']=='India']
    recovered_sum = recovered_country[cols[-1]]
    # calculate rates
    mortality_rate=round(float((death_sum/confirmed_sum))*100,2)
    recovery_rate=round(float((recovered_sum/confirmed_sum))*100,2)
    return mortality_rate,recovery_rate
    


