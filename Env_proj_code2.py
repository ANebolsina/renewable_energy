import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import geopandas as gpd
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
#data preprocessing
with st.echo(code_location='below'):
    plants = pd.read_csv('global_power_plant_database.csv')
    plants_geo = gpd.GeoDataFrame(plants, geometry = gpd.points_from_xy(plants['longitude'],plants['latitude']))
    plants_geo['coords']=list(zip(plants_geo['latitude'], plants_geo['longitude']))
    plants_geo.drop(plants_geo[plants_geo['primary_fuel']=='Storage'].index, inplace = True)


    urban = pd.read_csv('API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_2449640.csv', skiprows=4)
    urban.drop(['Unnamed: 65', '2020'], axis=1, inplace = True)
    urban.drop(urban[urban['1960'].isna()].index, inplace=True)

    temp = pd.read_csv('temp2.csv', sep = ';')
    temp[' Statistics'] = temp[' Statistics'].str[:4]

    en_consump = pd.read_csv('per-capita-energy-use.csv')


    st.set_page_config(page_title="Climate change explorer",
                      # page_icon='spotlog.png',
                       layout='wide')
    st.title('**Awareness of climate change trends**')
    st.markdown('**Disclaimer**: first, do not worry if it works slowly sometimes and your laptop sounds like helicopter - spatial plots and '
                 'data overall are quite massive, second, not to make you go through the whole code below the project I will put the technologies '
                 'that I have used after each block. Thanks! ')
    st.markdown("""
    This project have several **goals**:\n
    1) Provide a brief overview on climate change issue;\n
    2) Analyze several variables that are connected to the development of green energy;\n
    3) Build ML model that will predict the trend of renewable energy generation.\n
    Thus, there will be three parts. The first one is devoted to explain the phenomena of climate change and provide some facts about renewable energy. In the second 
    part there will be an illustration of factors that help to understand the trends in the development of renewable energy. The last part contains Machine learning model 
    that combines linear regression and random forest for predicition trends of energy generation by fuel types in Europe.
     """)

    st.header('**Overview of the theme**')
    st.markdown('First let us figure out what is the [difference between](https://climate.nasa.gov/resources/global-warming-vs-climate-change/) climate change and global warming that often could be used interchangeably. '
                'Global warming is the long-term heating of Earth’s climate system due to human activities. It was observed since the pre-industrial period'
                'Climate change is a long-term change in the average weather patterns that determine Earth’s climates. It can be caused both by human activities and natural'
                'processes. I will focus on human activity, in particular the issue of electricity generation, therefore we can touch upon both phenomena')
    st.subheader('**Sources of renewable energy**')

    st.markdown("Let us begin with taking a look of available source of renewable energy.")
    col1, col2, col3 = st.beta_columns(3)
    col1.subheader('Bioenergy')
    col1.write('Traditional use refers to the combustion of biomass in such forms as wood, animal waste '
             'and traditional charcoal. Modern bioenergy technologies include liquid biofuels produced f'
             'rom bagasse and other plants; bio-refineries, and other technologies.')
    col1.subheader('Solar')
    col1.write('There are two types of solar energy: *photovoltaics* (PV) are electronic devices that convert sunlight directly into electricity and'
               '*concentrated solar power* (CSP), that uses mirrors to concentrate solar rays. These rays heat fluid, which creates steam to drive a turbine and generate electricity.')
    col2.subheader('Geothermal')
    col2.write('Geothermal energy is heat derived within the sub-surface of the earth via water of stream. ')
    col2.subheader('Ocean')
    col2.write('Electricity is produced by Tides, waves and currents. ')
    col3.subheader('Hydropower')
    col3.write('Hydropower is energy derived from flowing water.')
    col3.subheader('Wind')
    col3.image('Fan.jpeg', use_column_width = True)


    st.subheader('**Current trends**')
    st.markdown('I have distinguished several factors that are interesting within the theme of renewable energy. We will explore some data on it'
                'and briefly discuss the connection to renewable energy development')
    st.markdown("""
    1) Economics of energy: production,  consumption, and costs;\n
    2) Urbanization level;\n
    3) Climate features;\n
    4) Governmental intervention; \n
    4) Social attitude; \n
    """)
    st.markdown('*Limitation*: we cannot fully explore this broad topic within this project, for example, we do not pay attention on technological development, '
                'patents, international agreements, etc. We just want to provide simple introductory tool that possibly will be developed to more'
                'complex and comprehensive project.  ')
    st.subheader('Available datasets')
    st.markdown('To begin, we should look on available datasets for our analysis')
    st.markdown('1. *Dataset on power plants*. Here we have data (2017) for all power plants by their type, location, capacity, and generation growth')
    st.write(pd.DataFrame(plants_geo[:5].drop('geometry', axis = 1)))
    st.markdown('2. *Dataset on level of urbanization*. It contains share of population that lives in cities. ')
    st.write(urban.head())
    st.markdown('3. *Dataset on average temperature*. Historical data with mounth temperature records')
    st.write(temp.head())
    st.markdown('4. *Dataset on energy consumption per capita*.')
    st.write(en_consump.head())
    costs = pd.read_excel('costs.xlsx',sheet_name='Лист1')
    st.markdown('4. *Dataset on energy price*.')
    st.write(costs.head())
    st.markdown('The rapid expansion of renewable energy could be explain by several reasons including dangers connected to global warming, '
                'limited amount of fossil fuels, but one of the most strong reason of its current success is the fact that **its price have become'
                ' comparable to fossil fuels**. On this graph we can see global weighted average cost of electricity (2019 USD/kWh) per type of renewable energy. Red line approximates current '
                'average price of electricity from gasoline/oil/coal.' )
    col = ['rgb(251,233,10)', 'rgb(141,221,236)', 'rgb(128,227,84)', 'rgb(230,145,56)']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=costs['Year'], y=costs['Concentrating solar power'],
                        mode='lines+markers',
                        name='Concentrating solar power',  marker_color=col[0]))
    fig.add_trace(go.Scatter(x=costs['Year'], y=costs['Offshore wind'],
                        mode='lines+markers',
                        name='Offshore wind',  marker_color=col[1]))
    fig.add_trace(go.Scatter(x=costs['Year'], y=costs['Onshore wind'],
                        mode='lines+markers',
                        name='Onshore wind',  marker_color=col[2]))
    fig.add_trace(go.Scatter(x=costs['Year'], y=costs['Solar photovoltaic'],
                        mode='lines+markers',
                        name='Solar photovoltaic',  marker_color=col[3]))
    fig.add_hline(y=0.08, line=dict(color="red",width=3))
    fig.update_layout(
        title="Global weighted average cost of electricity (2019 USD/kWh)",
        xaxis_title="Year",
        yaxis_title="USD/kWh",
        )
    st.write(fig)

    st.markdown('Here we take a look at plants distribution by their type around the world. Also, you can explore the number of different plants '
                'for each available country. '
                )
    colors=['lightgreen',] * 14
    fig = go.Figure(go.Bar(
                x=plants_geo.groupby('primary_fuel').count().sort_values(by = 'country',ascending=False )['country'].values[::-1],
                y=plants_geo.groupby('primary_fuel').count().sort_values(by = 'country',ascending=False )['country'].index[::-1],
                orientation='h', marker_color=colors))
    fig.update_layout(title_text='Total number of power plants')
    st.write(fig)
    country = st.selectbox(
         'Choose a country to explore', plants_geo['country_long'].unique())
    fig = go.Figure(go.Bar(
                x=plants_geo[plants_geo['country_long']==country].groupby('primary_fuel').count().sort_values(by = 'country', ascending=False)['country'].values[::-1],
                y=plants_geo[plants_geo['country_long']==country].groupby('primary_fuel').count().sort_values(by = 'country', ascending=False)['country'].index[::-1],
                orientation='h', marker_color=colors))
    fig.update_layout(title_text='Total number of power plants in {}'.format(country))
    st.write(fig)
    st.markdown('This pie chart shows the total capacity distribution by different fuels.')
    fig = px.pie(pd.DataFrame(plants_geo.groupby('primary_fuel').sum()['capacity_mw']).reset_index(),
                 values='capacity_mw', names='primary_fuel', title='Capacity distribution')
    st.write(fig)
    st.markdown('Now we will look only at how production capacity is distributed among different types of power plants.'
                ' Notice, that we have some big hydropower plants in China that prevent us from clear picture of the distribution,'
                'therefore we can omit this outliers by selecting plants with capacity below the  value from the slider.')

    border = st.radio('Select a capacity border', [100, 200, 300, 500, 1000, 1500, 2000, 5000, 10000, 22500])
    sup = plants_geo[plants_geo['capacity_mw'] <= border]
    fuels = plants_geo['primary_fuel'].unique()
    fig = go.Figure()
    for fuel in fuels:
        fig.add_trace(go.Violin(y=sup[sup['primary_fuel']==fuel]['capacity_mw'].values, name=fuel))
    st.write(fig)
    st.markdown('From this chart we can conclude that traditional fuel - oil, gas, nuclear - produce more energy, therefore we need less plants')
    st.markdown("Let us look on distribution at map: here by the color we distinguish plants by the fuel they use, and dots's size highlight the capacity of the plant")
    #fuel = st.selectbox(
     #    'Choose a primary fuel to explore the spatial distribution',
      #  plants_geo['primary_fuel'].unique())
    #m = folium.Map([25.75215, 37.61819], zoom_start=2)
    #for ind, row in plants_geo[plants_geo['primary_fuel']==fuel].iterrows():
     #   folium.Circle([row.latitude, row.longitude],
      #                radius=50).add_to(m)
    #folium_static(m)
    types = ['Hydro', 'Gas', 'Oil', 'Wind', 'Nuclear', 'Coal', 'Solar',
           'Waste', 'Biomass', 'Wave and Tidal', 'Petcoke', 'Geothermal',
           'Cogeneration', 'All']
    fuel_type = st.multiselect('Plants by fuel', types, 'All')
    if 'All' in fuel_type:
        fig2 = px.scatter_geo(plants_geo,
                       lat=plants_geo.geometry.y,
                       lon=plants_geo.geometry.x,
                       hover_name="name", size='capacity_mw', color='primary_fuel')
        st.write(fig2)
    else:
        pic = plants_geo[plants_geo['primary_fuel'].isin(fuel_type)]
        fig2 = px.scatter_geo(pic,
                             lat=pic.geometry.y,
                             lon=pic.geometry.x, size='capacity_mw', color='primary_fuel',
                             hover_name="name", hover_data=["primary_fuel", "capacity_mw", 'country_long'])
        st.write(fig2)



    st.markdown('Now, we will test whether it is true that the higher level of urbanization leads to '
                'higher consumption of energy. We will calculate the correlation between historic data of energy consumption and'
                'level of urbanization. We have calculated correlation for each country and plot the relation of variables. ')
    def get_consumption(name):
        return en_consump[en_consump['Entity']==name][['Year','Energy consumption per capita (kWh)']].set_index('Year')
    def get_corr(countcode):
        state = en_consump[en_consump['Code']==countcode]['Entity'].iloc[0]
        m= get_consumption(state)['Energy consumption per capita (kWh)'].index.min()
        M= get_consumption(state)['Energy consumption per capita (kWh)'].index.max()
        urb=urban[urban['Country Code']==countcode][range(m,M+1)].T[urban[urban['Country Code']==countcode].index[0]].values
        cons = get_consumption(state)['Energy consumption per capita (kWh)'].values
        return np.corrcoef(urb, cons)[0][1]
    dic = dict(zip(urban[urban['Country Code']=='AFG'].columns[4:],
                   list(urban[urban['Country Code']=='AFG'].columns[4:].astype(int))))
    urban.rename(columns=dic, inplace=True)
    missed = list(set(en_consump['Code'].unique()) - set(urban['Country Code'].unique()))
    for i in list(missed):
        en_consump.drop(en_consump[en_consump['Code']==missed[1]].index, inplace = True)
    common_countries = list(set(urban['Country Code'].unique()).intersection(set(en_consump['Code'].unique())))
    state = st.selectbox(
         'Choose a entity to explore the correlation between urbanization and energy consumption',
        en_consump[en_consump['Code'].isin(common_countries)]['Entity'].unique())
    entity = en_consump[en_consump['Entity']=='Afghanistan']['Code'][0]
    m = get_consumption(state)['Energy consumption per capita (kWh)'].index.min()
    M = get_consumption(state)['Energy consumption per capita (kWh)'].index.max()
    urb =urban[urban['Country Code']==entity][range(m,M+1)].T[urban[urban['Country Code']==entity].index[0]].values
    cons = get_consumption(state)['Energy consumption per capita (kWh)'].values
    fig = px.scatter(urb, cons, title="Scatter of urbanization level & energy consumption")
    fig.update_xaxes(title_text='Energy consumption')
    fig.update_yaxes(title_text='Urbanization level')
    st.write(fig)
    cor = {}
    for i in range(len(common_countries)):
        cor[urban[urban['Country Code']==common_countries[i]]['Country Name'].values[0]] = get_corr(common_countries[i])

    st.markdown('The correlation is {}'.format(cor[state]))
    st.markdown('Here you can explore the dimanic of average summer in winter temperatures in different countries.')
    c = st.selectbox(
         'Choose a country ',
        temp[' Country'].unique())
    t = temp.groupby(' Country').get_group(c)
    winter = [' Dec',' Jan', ' Feb']
    summer = [' Jun', ' Jul', ' Aug']

    t[t[' Statistics'].isin(winter)].groupby(' Year').mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[' Year'].unique(), y=t[t[' Statistics'].isin(winter)].groupby(' Year').mean()['Temperature - (Celsius)'],
                        mode='lines+markers',
                        name='Average winter temperature',
                            line=dict(color='darkblue')))
    fig.add_trace(go.Scatter(x=t[' Year'].unique(), y=t[t[' Statistics'].isin(summer)].groupby(' Year').mean()['Temperature - (Celsius)'],
                        mode='lines+markers',
                        name='Average summer temperature',
                            line=dict(color='yellow')))
    fig.update_layout(title='Average temperatures in {}'.format(c.strip()),
                       xaxis_title='Year',
                       yaxis_title='Temperature (Celsius)')
    st.write(fig)
    #st.markdown('We will add a variable that shows avergae annual growth of urbanization level.')
    #aavg = urban[range(1960,2020)].T.pct_change().sum(axis=0)/(2019-1960)
    st.markdown('We can calculate average annual growth rate for summer and winter temperature. ')
    winter = [' Dec',' Jan', ' Feb']
    summer = [' Jun', ' Jul', ' Aug']
    temp_sample = temp[temp[' Statistics'].isin(winter+summer)]
    def get_av_temp(country):
        auc = temp_sample[temp_sample[' Country']==country]
        t = len(auc[auc[' Statistics'].isin(winter)])/3
        wint = (auc[auc[' Statistics'].isin(winter)].groupby(' Year').mean().iloc[-1].values-auc[auc[' Statistics'].isin(winter)].groupby(' Year').mean()[:-1].median())/auc[auc[' Statistics'].isin(winter)].groupby(' Year').mean()[:-1].median()
        summ = (auc[auc[' Statistics'].isin(summer)].groupby(' Year').mean().iloc[-1].values-auc[auc[' Statistics'].isin(summer)].groupby(' Year').mean()[:-1].median())/auc[auc[' Statistics'].isin(summer)].groupby(' Year').mean()[:-1].median()
        return [wint[0], summ[0]]
    cagr_t={}
    for country in temp_sample[' Country'].unique():
        cagr_t[country.strip()]=get_av_temp(country)
    st.write(pd.DataFrame(cagr_t))

    st.markdown('Next, we will scrape national policies regarding to electricity and renewable energy that are currently in force in different countries'
                'from [policy database](https://www.iea.org/policies). We have applied such filters to pick up political iniiatives that are relevant'
                'to this project.')
    policies = []
    pages = list(range(1, 27))
    for i in pages:
        start_link = 'https://www.iea.org/policies?jurisdiction=National&sector=Electricity&status=In%20force&topic=Renewable%20Energy%2CClimate%20Change&page={}'.format(
            i)
        r = requests.get(start_link)
        if r.ok:
            soup = BeautifulSoup(r.text)
            for li in soup.find_all("li", {'class': 'm-policy-listing-item__row'}):
                policies.append(li)
    title = []
    country = []
    year = []
    for policy in policies:
        title.append(re.search('Climate%20Change">(.+)', str(policy.find_all('a')))[0][18:])
        country.append(re.search('data-sortable-key="country" data-sortable-value=(.+)', str(policy))[0][49:-2])
        year.append(re.search('data-sortable-key="year"(.+)', str(policy))[0][46:-2])
    energy_policy = pd.DataFrame({'Title': title, 'Country': country, 'Year': year})
    st.write(energy_policy.head())
    age = pd.Series(np.repeat(datetime.now().year, len(energy_policy['Year'])))-pd.to_datetime(energy_policy['Year']).dt.year
    energy_policy['Age']=age
    st.markdown('Here we can look at number of political measures and their median (as more robust measure of center) age.')
    colors2=['lightgreen',] * (len(energy_policy.groupby('Country').median()['Age']))
    fig = go.Figure([go.Bar(x=energy_policy.groupby('Country').count()['Title'].sort_values(ascending=False).index,
                            y=energy_policy.groupby('Country').count()['Title'].sort_values(ascending=False).values,
                            marker_color=colors2)])
    fig.update_layout(title_text='Total number of renewable energy political initiatives')
    st.write(fig)
    fig = go.Figure([go.Bar(x=energy_policy.groupby('Country').median()['Age'].sort_values(ascending=False).index,
                            y=energy_policy.groupby('Country').median()['Age'].sort_values(ascending=False).values,
                            marker_color=colors2)])
    fig.update_layout(title_text='Median age of renewable energy political initiatives')
    st.write(fig)
    st.markdown('For this data we can introduce the dummy variable that reflects is a policiy measure is relatively new or not. To do this we'
                'determine as a new the policies that are younger than half of all policies, i.e. its age is below the median')
    energy_policy['is_new']=energy_policy['Year'].copy()
    energy_policy.loc[energy_policy['Year'].astype(int) >= 2010, 'is_new']=1
    energy_policy.loc[energy_policy['Year'].astype(int) < 2010, 'is_new']=0
    fig = go.Figure(data=[
        go.Bar(name='New measures', x=energy_policy[energy_policy['is_new']==1]['Country'].value_counts().index,
               y=energy_policy[energy_policy['is_new']==1]['Country'].value_counts().values),
        go.Bar(name='Old measures',x=energy_policy[energy_policy['is_new']==0]['Country'].value_counts().index,
               y=energy_policy[energy_policy['is_new']==0]['Country'].value_counts().values)])

    fig.update_layout(barmode='group')
    st.write(fig)
    st.markdown('*Used technologies*: Pandas, web scraping, data visualisation, mathematics with numpy, streamlit, regular expressiohs, geopandas / follium.')
    st.header('**Prognosis and model**')
    st.markdown('In this section we  build a model that predicts energy generation by country and type of fuel. We take this dataset '
                'with monthly energy generation by fuel type in European countries from 2018 till present.')

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    enprod = pd.read_excel('Ember-EU-Electricity-Data-May-2021.xlsx', sheet_name = 'Data')
    variable_of_interest = ['Fossil', 'Renewables', 'Nuclear']
    df = enprod[enprod['Variable.Type'].isin(variable_of_interest)] #final dataset
    df.rename(columns={"Variable": "Fuel", "Variable.Type": "FuelType"}, inplace = True)
    st.write(df.head())
    st.write('Now we will prepare data for training: we need to normalize time variables and encode categorical variables.'
             'We will focus on date, area, month, fueltype, and weekday to predict generation ')
    features = ['Date', 'Area', 'FuelType', 'Year', 'Month']
    target ='Generation (GWh)'
    X = df[features]
    y = df[target].values
    X['weekday'] = X['Date'].dt.weekday
    X['norm_time']=(((X['Year'] - 2018) * 12 + X['Month'])-((X['Year'] - 2018) * 12 + X['Month']).mean())/((X['Year'] - 2018) * 12 + X['Month']).std()
    X.drop('Month', axis = 1, inplace = True)
    X_train = X[X['Date'].dt.year<=2020]
    X_test = X[X['Date'].dt.year>2020]
    y_train = y[:7996]
    y_test=y[7996:]
    st.markdown("As train set we took data before 2020, and as test one - after 2020.")
    X_train.drop(['Date', 'Year'], axis =1, inplace = True)
    X_test.drop(['Date', 'Year'], axis =1, inplace = True)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(X_train[['Area', 'FuelType']])
    xtr = pd.DataFrame(enc.transform(X_train[[ 'Area','FuelType']]))
    X_train2 = pd.DataFrame(X_train['norm_time']).reset_index(drop=True).merge(xtr, left_index=True, right_index=True)
    xts = pd.DataFrame(enc.transform(X_test[[ 'Area','FuelType']]))
    X_test2 = pd.DataFrame(X_test['norm_time']).reset_index(drop=True).merge(xts, left_index=True, right_index=True)
    st.markdown("There is final train set.")
    st.write(X_train2.head())
    st.image('prepmem.jpeg')
    st.markdown("Then we train Ridge regression with cross validation to choose the best alpha, but get the negative score that basically reflects"
                "the inefficiency of the model :(")
    regr_cv = RidgeCV(alphas=[0.01, 0.1, 0.2, 0.5, 0.7, 1.0,2.0, 5.0, 7.0])
    model_cv = regr_cv.fit(xtr, y_train)
    st.write('Optimal alpha {}'.format(model_cv.alpha_))
    st.write('Model score {}'.format(model_cv.score(xts, y_test)))
    st.markdown("After we try to fit pure data and regression's results to Random Forest model (try several depth levels), but still results are unsatisfactory.")
    regr4 = RandomForestRegressor(max_depth=6)
    regr4.fit(X_train2, y_train)
    st.write('Random forest score {}'.format(regr4.score(X_test2, y_test)))

    st.markdown('Such results can be explained by the small number of observations and luck of numerical variables,'
                'we have only categorical variables. ')
    st.image('MLmeme.jpeg')
    st.write('But we did not give up and move to *R* where we have biult the special [model](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model)'
                ' for time series predictions as well as plot some cool graphics. The choice of R is explained by the fact that I do not know such'
             'convenient libraries for such modeling in Python.')

    st.markdown('Here you can find [R code](http://rpubs.com/env_rpoject_BAE_DS/r_part) (It also could be opened from archive). Main results and plots will be below. We will include plots'
                'for Germany, Netherlands, and Finland, but in script you can find replication for each country.')
    st.markdown('First, we plot (by ggplot2 extensions) the distribution of energy generation by fuel type for each European country in the sample')
    st.markdown('Second, we build ARIMA model to predict energy generation for renewables and fuels for each country. In the graph you can find prognosis for 5 month'
                'forward with 80% (dark blue) and 90% (light blue) confidence intervals. After we plot prediction for both fossils in renewables in the one plot with several layers')
    st.markdown('*Used technologies*: Machine Learning, statistical model, R, tidyverse (data manipulation and plots),'
                'ggplot2 visualisation (including several layers), ggplot2 extensions.')
    st.subheader('**Germany**')
    st.image('germ_dens.jpeg')
    st.markdown('Predictions for fossils')
    st.image('germfossil.jpeg')
    st.markdown('Predictions for renewables')
    st.image('gerrenew.jpeg')
    st.image('germprog.jpeg')
    st.markdown('Same exercise for Netherlands. ')
    st.image('nethptog.jpeg')
    st.markdown('Also there is a trends for Finland.')
    st.image('finprog.jpeg')
    st.markdown('Initially I have chosen Sweden for showing the results, but it turns out that they do not have data on fossils that reflects'
                'huge progress in terms of green energetic!')
    st.header('**Conclusions and observations**')

    st.markdown("""
    - We  provide an overview of climate change and existing sources for renewable energy,\n
    - We show that there is a positive dynamic in number of plants basing on renewables that can be explained by decreased costs,\n
    - We illustrate the capacity distribution: plants with renewable sources have smaller capacity, therefore we need more of them, \n
    - We shaw that overall there is positive correlation between urbanization level and energy consumption that means the need for more energy generation.\n
    - There are positive trend in temperature level, thus the care of reducing CO-2 emissions by switching to renewables is reasonable,\n
    - Now countries take more (direct, in our case) initiatives on issues on renewable energy (there more relativle new measures) that manifistates the urgency of the issue and 
    need for government involvement,\n
    - Model predictions show that time cyclicality of generation will persist but we see the positive trend for renewable energy generation in European countries. """)
    with st.beta_expander("Data sources"):
        st.markdown('- [Power plants](https://datasets.wri.org/dataset/globalpowerplantdatabase) \n'
                    '- [Temperature](https://climateknowledgeportal.worldbank.org/download-data) \n'
                    '- [Energy costs](https://www.irena.org/publications/2020/Jun/Renewable-Power-Costs-in-2019)\n'
                    '- [Political initiatives](https://www.iea.org/policies) \n'
                    '- [Energy generation in EU](https://ember-climate.org/european-electricity-transition/)\n')

