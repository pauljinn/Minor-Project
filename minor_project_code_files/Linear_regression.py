import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def return_dataframe():
    df = pd.read_csv("Air Quality123.csv")
    return df
def data_trainer(df):
    x = df[['PM2.5','PM10','NO2','NH3','SO2','CO','OZONE']]
    y = df['AirQualityIndex']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    #random_state=10
    reg = linear_model.LinearRegression()
    reg.fit(x_train,y_train)
    return reg,x_test,y_test
def print_accuracy():
    df = return_dataframe()
    reg,x_test,y_test = data_trainer(df)
    print(reg.predict(x_test))
    print(y_test)
    print(reg.score(x_test,y_test)*100,"%")
    print(x_test)
def user_aqi_predictor(PM25,PM10,NO2,NH3,SO2,CO,OZONE):
    df = return_dataframe()
    reg,x_test,y_test = data_trainer(df)
    aqi_value = reg.predict([[PM25,PM10,NO2,NH3,SO2,CO,OZONE]])
    return aqi_value
def pm25_scatter_plot():
    df = return_dataframe()
    plt.scatter(df['PM2.5'],df['AirQualityIndex'],s=2)
    plt.title("Scatter Plot 1")
    plt.xlabel('PM2.5')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.show()
def pm10_scatter_plot():
    df = return_dataframe()
    plt.scatter(df['PM10'],df['AirQualityIndex'],s=1)
    plt.title("Scatter Plot 2")
    plt.xlabel('PM10')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.show()
def no2_scatter_plot():
    df = return_dataframe()
    plt.scatter(df['NO2'],df['AirQualityIndex'],s=1)
    plt.title("Scatter Plot 3")
    plt.xlabel('NO2')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.show()
def nh3_scatter_plot():
    df = return_dataframe()
    plt.scatter(df['NH3'],df['AirQualityIndex'],s=0.5)
    plt.title("Scatter Plot 4")
    plt.xlabel('NH3')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.show()
def so2_scatter_plot():
    df = return_dataframe()
    plt.scatter(df['SO2'],df['AirQualityIndex'],s=1)
    plt.title("Scatter Plot 5")
    plt.xlabel('SO2')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.show()
def co_scatter_plot():
    df = return_dataframe()
    plt.scatter(df['CO'],df['AirQualityIndex'],s=1)
    plt.title("Scatter Plot 6")
    plt.xlabel('CO')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.show()
def ozone_scatter_plot():
    df = return_dataframe()
    plt.scatter(df['OZONE'],df['AirQualityIndex'],s=1)
    plt.title("Scatter Plot 7 ")
    plt.xlabel('OZONE')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.show()
def parameters_scatter_plot():
    df = return_dataframe()
    plt.scatter(df['PM2.5'],df['AirQualityIndex'],s=0.3)
    plt.scatter(df['PM10'],df['AirQualityIndex'],s =0.3,color = 'indigo')
    plt.scatter(df['NO2'],df['AirQualityIndex'],s=0.3,color = 'blue')
    plt.scatter(df['NH3'],df['AirQualityIndex'],s=0.3,color='green')
    plt.scatter(df['SO2'],df['AirQualityIndex'],s=0.3,color='yellow')
    plt.scatter(df['CO'],df['AirQualityIndex'],s=0.3,color='orange')
    plt.scatter(df['OZONE'],df['AirQualityIndex'],s=0.3,color='red')
    plt.title("Scatter Plot 8")
    plt.xlabel('PM2.5,PM10,NO2,NH3,SO2,CO,OZONE')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.show()
def classify_air_quality(aqi_value):
    if(aqi_value in range(0,51)):
        quality = "Good"
        message = health_messages(quality)
        return quality,message
    elif(aqi_value in range(51,101)):
        quality = "Satisfactory"
        message = health_messages(quality)
        return quality,message
    elif(aqi_value in range(101,201)):
        quality = "Moderately polluted"
        message = health_messages(quality)
        return quality,message
    elif(aqi_value in range(201,301)):
        quality = "Poor"
        message = health_messages(quality)
        return quality,message
    elif(aqi_value in range(301,401)):
        quality = "Very poor"
        print(quality)
        message = health_messages(quality)
        return quality,message
    elif(aqi_value>=401):
        quality = "Severe"
        message = health_messages(quality)
        return quality,message
    '''
    else:
        print("The air quality value is not legal")
    '''
def health_messages(quality):
    if(quality=="Good"):
        message = "Minimal Impact"
        return message
    elif(quality=="Satisfactory"):
        message = "May cause minor breathing discomfort to sensitive people."
        return message 
    elif(quality=="Moderately polluted"):
         message = "May cause breathing discomfort to people with lung disease such as asthma and discomfort to people with heart disease, children and older adults."
         return message
    elif(quality=="Poor"):
        message = "May cause breathing discomfort to people on prolonged exposure and discomfort to people with heart disease."
        return message
    elif(quality=="Very poor"):
        message = "May cause respiratory illness to the people on prolonged exposure. Effect may be more pronounced in people with lung and heart diseases."
        return message
    elif(quality=="Severe"):
         message = "May cause respiratory impact even on healthy people, and serious health impacts on people with lung/heart disease. The health impacts may be experienced even during light physical activity."
         return message
def call_functions(aqi):
    return classify_air_quality(int(aqi))
    
