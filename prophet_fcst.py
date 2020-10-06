#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 18:00:20 2020

@author: David.Hidasi
"""

import pandas as pd
import numpy as np
import multiprocessing as mp
import itertools
from fbprophet import Prophet
import datetime, time
import dateutil.parser
import inspect
import warnings
import copy


def minutes_from_epoch(ds):
    return [time.mktime(x.timetuple()) for x in ds]

# this strips h:m:s as well
def remove_timezone(ts):
    ts = dateutil.parser.parse(ts)
    return(ts.strftime('%m/%d/%Y'))

def log_result(result):
    result_list.append(result)
    
def days_between(d1, d2):
    if isinstance(d1, datetime.datetime) == False:
        d1 = datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days);

def data_prep(DF, ticker, horizon, shift = 0, cv_sets = 0):    
    df = copy.deepcopy(DF)
    df = pd.DataFrame(df[ticker])#[:-1]
    df['ds'] = df.index
    
    non_target_cols = list([a for a in df.columns if a != 'ds' and a != 'Close']) 
    df = df.reindex(columns=(['ds', 'Close'] + non_target_cols))
    df = df.rename(columns={"Close": "y"})
    
    if shift > 0:
        regressor_cols = list([a for a in non_target_cols if a not in ('Open','High','Low', 'Volume', 'Date','ret')])
        for col in regressor_cols:
            df[col] = df[col].shift(shift);
    
    
    df['cv_set' + str(0)] = 'train'
    df['cv_set' + str(0)][-horizon:] = np.array('test')
    
    if cv_sets >0:
        for cv_set in range(1,cv_sets):
            df['cv_set' + str(cv_set)] = df['cv_set0'].shift(-cv_set)
    
    # if weekends are skipped for in fxdata, make sure that the fcst length covers weekdays
    days_diff = days_between(min(df[df['cv_set0'] == 'test']['ds']), 
                             max(df[df['cv_set0'] == 'test']['ds'])) # .strftime('%Y-%m-%d')
    
                   
    if days_diff > horizon:
        horizon = days_diff;
        
    return(df, horizon);
    



def prophet_estimate(prophet_training_data, 
                     prophet_test_data,
                     horizon,
                   # holidays,
                    changepoints_pr,
                    holiday_pr,
                    seasonality_m,
                    seasonality_pr,
                    yearly_seas,
                    weekly_seas,
                    daily_seas,
                    regressors = [],
                    regressor_pr = None,
                    tune = True,
                    verbose = False,
                    freq = 'D',
                    include_hist = False):
    m = Prophet(#holidays=holidays, 
                        changepoint_prior_scale=changepoints_pr, 
                        holidays_prior_scale=holiday_pr, 
                        seasonality_mode= seasonality_m,
                        weekly_seasonality=weekly_seas, 
                        yearly_seasonality = yearly_seas,
                        seasonality_prior_scale=seasonality_pr, 
                        daily_seasonality=daily_seas)
    
    training_data = copy.deepcopy(prophet_training_data)
    test_data = copy.deepcopy(prophet_test_data)
#    ,
    # Add each regressor (if any) to the model (before fitting)
    if len(regressors) > 0:
        # drop rows with na in regressor cols
        training_data.dropna(subset=regressors, inplace = True)
        
        for regressor in regressors:
            m.add_regressor(
                            name = regressor,
                            standardize = 'auto',
                            prior_scale = regressor_pr
                            )                  
    # Fit model
    fit_model = m.fit(training_data)
    # If there are no regressors, create a future df simply based on dates
    future = fit_model.make_future_dataframe(periods=horizon, 
                                             freq=freq, #'2 min',
                                             include_history=include_hist)

    # If regressors are to be used then add regressor columns from test to future df
    if len(regressors) > 0:
        for regressor in regressors:
            if include_hist == False:
                future[regressor] = np.array(test_data[regressor])
            else:
                future[regressor] = training_data[regressor].append(test_data[regressor],
                      ignore_index=True)

    fcst = fit_model.predict(future)
    if include_hist == False:
        joined_data = pd.merge(fcst, test_data, how='inner', on=['ds'])
    else:
        joined_data = pd.merge(fcst, training_data.append(test_data), how='left', on=['ds'])
    rep_obj = {'changepoints_pr' : changepoints_pr,
    'holiday_pr' : holiday_pr, 
    'seasonality_m' : seasonality_m,
    'seasonality_pr' : seasonality_pr, 
    'yearly_seas' : yearly_seas,
    'weekly_seas' : weekly_seas,
    'daily_seas' : daily_seas,
    'regressors' : regressors, #','.join(regressors)
    'regressor_pr' : regressor_pr,
    'mape' :np.mean([(abs(x - y)/x) for (x,y) in zip(joined_data.y, joined_data.yhat)]),
    'mae' :np.mean([abs(x - y) for (x,y) in zip(joined_data.y, joined_data.yhat)]) };
    if tune == False:
        rep_obj = joined_data
    if verbose == True: 
        print(rep_obj);
    return(rep_obj);