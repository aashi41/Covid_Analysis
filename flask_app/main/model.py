import datetime
import numpy as np 
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import pandas as pd
import matplotlib.pyplot as plt 
from . import covid_info
#import covid_info


cols = covid_info.cols
confirmed = covid_info.confirmed_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys() #date value present as DF columns
days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1) #getting total no of days as serial no like 1,2,3, till next future 10 days
adjusted_dates = future_forcast[:-10] # remove future 10 dates as a serial no
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y') 
future_forcast_dates = [] #to get dates from 22 jan 2020 to future next 10 days
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

confirmed = covid_info.confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = covid_info.death_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    
    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)

    # calculate rates
    mortality_rate.append(death_sum/confirmed_sum)


#to get best degree for polynomial
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1) #get dates as serial no like 1,2,3
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
#slightly modify the data to fit the model better (regression models cannot pick the pattern)
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[:], world_cases[:], test_size=0.03, shuffle=False) 




#Polynomial Regresssion
def Linear_model_training():
    rmses = []
    degrees = np.arange(1,10)
    min_rmse, min_deg = 1e10, 0

    for deg in degrees:
        
        poly = PolynomialFeatures(degree=deg)
        poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
        poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
        poly_future_forcast = poly.fit_transform(future_forcast)
        linear_model = LinearRegression(normalize=True, fit_intercept=False)
        linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
        test_linear_pred = linear_model.predict(poly_X_test_confirmed)
        linear_pred = linear_model.predict(poly_future_forcast)
        poly_mse = mean_squared_error(test_linear_pred, y_test_confirmed)
        poly_rmse = np.sqrt(poly_mse)
        rmses.append(poly_rmse)
        print(f"{deg} and {poly_rmse}")
        
        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg
            

    print(f"best degree is {min_deg} with RMSE {min_rmse}")



    # transform our data for polynomial regression (using degree as 6 as it is giving us min mse - check code below few cells)
    poly = PolynomialFeatures(degree=min_deg)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)


    # polynomial regression
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)
    print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
    print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
    print('r2:',r2_score(y_test_confirmed,test_linear_pred))


    # Future predictions using polynomial regression
    linear_pred = linear_pred.reshape(1,-1)[0]
    linear_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Polynomial Predicted # of Confirmed Cases Worldwide': np.round(linear_pred[-10:])})
    linear_df.style.background_gradient(cmap='Reds')
    print(linear_df)

    pickle.dump(linear_model, open('linear_model.pkl','wb'))
    #plt.plot(y_test_confirmed)
    #plt.plot(test_linear_pred)
    #plt.legend(['Test Data', 'Polynomial Regression Predictions'])
    #plt.show()

#Linear_model_training()


def ploy_predict():
    poly = PolynomialFeatures(degree=3)
    poly_future_forcast = poly.fit_transform(future_forcast)
    #model = pickle.load(open('linear_model.pkl','rb'))
    model = pickle.load(open('D:\\My_study_for_data_science\\python\\covid_detection\\covid_detection\\covid_detection\\flask_app\\main\\linear_model.pkl','rb'))
    linear_pred = model.predict(poly_future_forcast)
    linear_pred = linear_pred.reshape(1,-1)[0]
    linear_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Polynomial Predicted No. of Confirmed Cases Worldwide': np.round(linear_pred[-10:])})
    linear_df.style.background_gradient(cmap='Reds')
    print(linear_df)
    predicted_list = [linear_df.columns.values.tolist()] + linear_df.values.tolist()
    print(predicted_list)
    return predicted_list

    

ploy_predict()

