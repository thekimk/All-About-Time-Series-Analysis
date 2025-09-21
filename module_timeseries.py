# Ignore the warnings
import warnings
# warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# System related and data input controls
import os

# Data manipulation and visualization
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_rows = 20
pd.options.display.max_columns = 20
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from tqdm import tqdm

# Modeling algorithms
# General
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# Model selection
from sklearn.model_selection import train_test_split

# Evaluation metrics
from sklearn import metrics
# for regression
from sklearn.metrics import mean_squared_error,  mean_absolute_error, mean_absolute_percentage_error



# 시계열 변수추출
## 날짜인식 및 빈도 설정만
def non_feature_engineering(df):
    df_nfe = df.copy()
    if 'datetime' in df_nfe.columns:
        df_nfe['datetime'] = pd.to_datetime(df_nfe['datetime'])
        df_nfe['DateTime'] = pd.to_datetime(df_nfe['datetime'])
    if df_nfe.index.dtype == 'int64':
        df_nfe.set_index('DateTime', inplace=True)
    df_nfe = df_nfe.asfreq('H', method='ffill')

    return df_nfe


## 날짜인식 및 빈도 설정을 포함한 모든 전처리
def feature_engineering(df):
    df_fe = df.copy()
    if 'datetime' in df_fe.columns:
        df_fe['datetime'] = pd.to_datetime(df_fe['datetime'])
        df_fe['DateTime'] = pd.to_datetime(df_fe['datetime'])

    if df_fe.index.dtype == 'int64':
        df_fe.set_index('DateTime', inplace=True)

    df_fe = df_fe.asfreq('H', method='ffill')

    result = sm.tsa.seasonal_decompose(df_fe['count'], model='additive')
    Y_trend = pd.DataFrame(result.trend)
    Y_trend.fillna(method='ffill', inplace=True)
    Y_trend.fillna(method='bfill', inplace=True)
    Y_trend.columns = ['count_trend']
    Y_seasonal = pd.DataFrame(result.seasonal)
    Y_seasonal.fillna(method='ffill', inplace=True)
    Y_seasonal.fillna(method='bfill', inplace=True)
    Y_seasonal.columns = ['count_seasonal']
    pd.concat([df_fe, Y_trend, Y_seasonal], axis=1).isnull().sum()
    if 'count_trend' not in df_fe.columns:
        if 'count_seasonal' not in df_fe.columns:
            df_fe = pd.concat([df_fe, Y_trend, Y_seasonal], axis=1)

    Y_count_Day = df_fe[['count']].rolling(24).mean()
    Y_count_Day.fillna(method='ffill', inplace=True)
    Y_count_Day.fillna(method='bfill', inplace=True)
    Y_count_Day.columns = ['count_Day']
    Y_count_Week = df_fe[['count']].rolling(24*7).mean()
    Y_count_Week.fillna(method='ffill', inplace=True)
    Y_count_Week.fillna(method='bfill', inplace=True)
    Y_count_Week.columns = ['count_Week']
    if 'count_Day' not in df_fe.columns:
        df_fe = pd.concat([df_fe, Y_count_Day], axis=1)
    if 'count_Week' not in df_fe.columns:
        df_fe = pd.concat([df_fe, Y_count_Week], axis=1)

    Y_diff = df_fe[['count']].diff()
    Y_diff.fillna(method='ffill', inplace=True)
    Y_diff.fillna(method='bfill', inplace=True)
    Y_diff.columns = ['count_diff']
    if 'count_diff' not in df_fe.columns:
        df_fe = pd.concat([df_fe, Y_diff], axis=1)

#     df_fe['temp_group'] = pd.cut(df_fe['temp'], 10)
    df_fe['Year'] = df_fe.datetime.dt.year
    df_fe['Quater'] = df_fe.datetime.dt.quarter
    df_fe['Quater_ver2'] = df_fe['Quater'] + (df_fe.Year - df_fe.Year.min()) * 4
    df_fe['Month'] = df_fe.datetime.dt.month
    df_fe['Day'] = df_fe.datetime.dt.day
    df_fe['Hour'] = df_fe.datetime.dt.hour
    df_fe['DayofWeek'] = df_fe.datetime.dt.dayofweek

    df_fe['count_lag1'] = df_fe['count'].shift(1)
    df_fe['count_lag2'] = df_fe['count'].shift(2)
    df_fe['count_lag1'].fillna(method='bfill', inplace=True)
    df_fe['count_lag2'].fillna(method='bfill', inplace=True)

    if 'Quater' in df_fe.columns:
        if 'Quater_Dummy' not in ['_'.join(col.split('_')[:2]) for col in df_fe.columns]:
            df_fe = pd.concat([df_fe, pd.get_dummies(df_fe['Quater'], 
                                                     prefix='Quater_Dummy', drop_first=True).astype(int)], axis=1)
            del df_fe['Quater']
    
    return df_fe


##################################################
# 2011년 데이터 시계열 패턴으로 2012년 데이터로 가정
def feature_engineering_year_duplicated(X_train, X_test, target):
    X_train_R, X_test_R = X_train.copy(), X_test.copy()
    for col in target:
        X_train_R.loc['2012-01-01':'2012-02-28', col] = X_train_R.loc['2011-01-01':'2011-02-28', col].values
        X_train_R.loc['2012-03-01':'2012-06-30', col] = X_train_R.loc['2011-03-01':'2011-06-30', col].values
        X_test_R.loc['2012-07-01':'2012-12-31', col] = X_train_R.loc['2011-07-01':'2011-12-31', col].values
        
        step = (X_train_R.loc['2011-03-01 00:00:00', col] - X_train_R.loc['2011-02-28 23:00:00', col])/25
        step_value = np.arange(X_train_R.loc['2011-02-28 23:00:00', col]+step, 
                               X_train_R.loc['2011-03-01 00:00:00', col], step)
        step_value = step_value[:24]
        X_train_R.loc['2012-02-29', col] = step_value

    return X_train_R, X_test_R


# 종속변수에서 지연처리 후 NaN은 Train이 아닌 Test 값에서 채움
def feature_engineering_lag_modified(Y_test, X_test, target):
    X_test_R = X_test.copy()
    for col in target:
        X_test_R[col] = Y_test.shift(1).values
        X_test_R[col].fillna(method='bfill', inplace=True)
        X_test_R[col] = Y_test.shift(2).values
        X_test_R[col].fillna(method='bfill', inplace=True)
        
    return X_test_R


# 스케일 조정
def feature_engineering_scaling(scaler, X_train, X_test):
    # preprocessing.MinMaxScaler(), preprocessing.StandardScaler(), preprocessing.RobustScaler(), preprocessing.Normalizer()
    scaler = scaler
    scaler_fit = scaler.fit(X_train)
    X_train_scaling = pd.DataFrame(scaler_fit.transform(X_train), 
                               index=X_train.index, columns=X_train.columns)
    X_test_scaling = pd.DataFrame(scaler_fit.transform(X_test), 
                               index=X_test.index, columns=X_test.columns)
    
    return X_train_scaling, X_test_scaling


# 모든 독립변수들의 VIF 수치 확인 및 오름차순 정렬 후 상위 num_variables개 독립변수만 추출
def feature_engineering_XbyVIF(X_train, num_variables):
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [variance_inflation_factor(X_train.values, i) 
                         for i in range(X_train.shape[1])]
    vif['Feature'] = X_train.columns
    X_colname_vif = vif.sort_values(by='VIF_Factor', ascending=True)['Feature'][:num_variables].values
    
    return X_colname_vif
##################################################


# 데이터 분리
## cross sectional 데이터
def datasplit(df, Y_colname, test_size=0.2, random_state=123):
    X_colname = [x for x in df.columns if x not in Y_colname]
       
    X_train, X_test, Y_train, Y_test = train_test_split(df[X_colname], df[Y_colname],
                                                        test_size=test_size, random_state=random_state)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    
    return X_train, X_test, Y_train, Y_test


## time series 데이터
def datasplit_ts(df, Y_colname, X_colname, criteria):
    df_train = df.loc[df.index < criteria,:]
    df_test = df.loc[df.index >= criteria,:]
    Y_train = df_train[Y_colname]
    X_train = df_train[X_colname]
    Y_test = df_test[Y_colname]
    X_test = df_test[X_colname]
    print('Train_size:', df_train.shape, 'Test_size:', df_test.shape)
    print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
    print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)
    
    return X_train, X_test, Y_train, Y_test


# 실제 Y와 예측치 시각화
def plot_prediction(Y_true_pred):
    plt.figure(figsize=(16, 8))
    plt.plot(Y_true_pred, linewidth=5, label=Y_true_pred.columns)
    plt.xticks(fontsize=25, rotation=0)
    plt.yticks(fontsize=25)
    plt.xlabel('Index', fontname='serif', fontsize=28)
    plt.legend(fontsize=20)
    plt.grid()
    plt.show()
    
    
# 검증 함수화
def evaluation_reg(Y_real, Y_pred):
    MAE = mean_absolute_error(Y_real, Y_pred)
    MSE = mean_squared_error(Y_real, Y_pred)
    MAPE = mean_absolute_percentage_error(Y_real, Y_pred)
    Score = pd.DataFrame([MAE, MSE, MAPE], index=['MAE', 'MSE', 'MAPE'], columns=['Score']).T
    
    return Score

# Train & Test 모두의 검증 함수화
def evaluation_reg_trte(Y_real_tr, Y_pred_tr, Y_real_te, Y_pred_te):
    Score_tr = evaluation_reg(Y_real_tr, Y_pred_tr)
    Score_te = evaluation_reg(Y_real_te, Y_pred_te)
    Score_trte = pd.concat([Score_tr, Score_te], axis=0)
    Score_trte.index = ['Train', 'Test']

    return Score_trte
    # Setting
    Resid = Residual.copy()
    if Resid.shape[0] >= 100:
        lag_max = 50
    else:
        lag_max = int(Resid.shape[0]/2)-1

# 에러 분석
def error_analysis_timeseries(X_Data, Y_Pred, Residual, graph_on=False):
    # Setting
    Resid = Residual.copy()
    if Resid.shape[0] >= 100:
        lag_max = 50
    else:
        lag_max = int(Resid.shape[0]/2)-1
        
    if graph_on == True:
        ##### 시각화
        # index를 별도 변수로 저장 
        Resid = Residual.copy()
        Resid['Index'] = Resid.reset_index().index
    
        # 잔차의 정규분포성 확인
        sns.distplot(Resid.iloc[:,[0]], norm_hist='True', fit=stats.norm)
        plt.show()

        # 잔차의 등분산성 확인
        sns.lmplot(data=Resid, x='Index', y=Resid.columns[0],
                   fit_reg=True, line_kws={'color': 'red'}, height=5, aspect=2, ci=99, sharey=True)
        plt.show()
        
        # 잔차의 자기상관성 확인
        sm.graphics.tsa.plot_acf(Resid.iloc[:,[0]], lags=lag_max, use_vlines=True)
        plt.ylabel('Correlation')
        plt.show()
        
        # 잔차의 편자기상관성 확인
        sm.graphics.tsa.plot_pacf(Resid.iloc[:,[0]], lags=lag_max, use_vlines=True)
        plt.ylabel('Correlation')
        plt.show()

    ##### 통계량
    # 정규분포
    # Null Hypothesis: The residuals are normally distributed
    Normality = pd.DataFrame([stats.shapiro(Residual)], 
                             index=['Normality'], columns=['Test Statistics', 'p-value']).T

    # 등분산성
    # Null Hypothesis: Error terms are homoscedastic
    Heteroscedasticity = pd.DataFrame([sm.stats.diagnostic.het_goldfeldquandt(Residual, X_Data.values, alternative='two-sided')],
                                      index=['Heteroscedasticity'], 
                                      columns=['Test Statistics', 'p-value', 'Alternative']).T
    
    # 자기상관
    # Null Hypothesis: Autocorrelation is absent
    Autocorrelation = pd.concat([pd.DataFrame(sm.stats.diagnostic.acorr_ljungbox(Residual, lags=[10,lag_max]).iloc[:,0]),
                             pd.DataFrame(sm.stats.diagnostic.acorr_ljungbox(Residual, lags=[10,lag_max]).iloc[:,1])], axis=1).T
    Autocorrelation.index = ['Test Statistics', 'p-value']
    Autocorrelation.columns = ['Autocorr(lag10)', 'Autocorr(lag50)']
    
    # 정상성
    # ADF
    # Null Hypothesis: The Time-series is non-stationalry
    Stationarity = pd.Series(sm.tsa.stattools.adfuller(Residual)[0:3], 
                             index=['Test Statistics', 'p-value', 'Used Lag'])
    for key, value in sm.tsa.stattools.adfuller(Resid.iloc[:,[0]])[4].items():
        Stationarity['Critical Value(%s)'%key] = value
    Stationarity_ADF = pd.DataFrame(Stationarity, columns=['Stationarity_ADF'])
    # KPSS
    # Null Hypothesis: The Time-series is stationalry
    Stationarity = pd.Series(sm.tsa.stattools.kpss(Residual)[0:3], 
                             index=['Test Statistics', 'p-value', 'Used Lag'])
    for key, value in sm.tsa.stattools.kpss(Resid.Error)[3].items():
        if key != '2.5%':
            Stationarity['Critical Value(%s)'%key] = value
    Stationarity_KPSS = pd.DataFrame(Stationarity, columns=['Stationarity_KPSS'])
    
    Error_Analysis = pd.concat([Normality, Heteroscedasticity, Autocorrelation,
                                Stationarity_ADF, Stationarity_KPSS], join='outer', axis=1)
    
    return Error_Analysis


def stationarity_adf_test(Y_Data, Target_name):
    if len(Target_name) == 0:
        Stationarity_adf = pd.Series(sm.tsa.stattools.adfuller(Y_Data)[0:4],
                                     index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
        for key, value in sm.tsa.stattools.adfuller(Y_Data)[4].items():
            Stationarity_adf['Critical Value(%s)'%key] = value
            Stationarity_adf['Maximum Information Criteria'] = sm.tsa.stattools.adfuller(Y_Data)[5]
            Stationarity_adf = pd.DataFrame(Stationarity_adf, columns=['Stationarity_adf'])
    else:
        Stationarity_adf = pd.Series(sm.tsa.stattools.adfuller(Y_Data[Target_name])[0:4],
                                     index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
        for key, value in sm.tsa.stattools.adfuller(Y_Data[Target_name])[4].items():
            Stationarity_adf['Critical Value(%s)'%key] = value
            Stationarity_adf['Maximum Information Criteria'] = sm.tsa.stattools.adfuller(Y_Data[Target_name])[5]
            Stationarity_adf = pd.DataFrame(Stationarity_adf, columns=['Stationarity_adf'])
    return Stationarity_adf

def stationarity_kpss_test(Y_Data, Target_name):
    if len(Target_name) == 0:
        Stationarity_kpss = pd.Series(sm.tsa.stattools.kpss(Y_Data)[0:3],
                                      index=['Test Statistics', 'p-value', 'Used Lag'])
        for key, value in sm.tsa.stattools.kpss(Y_Data)[3].items():
            Stationarity_kpss['Critical Value(%s)'%key] = value
            Stationarity_kpss = pd.DataFrame(Stationarity_kpss, columns=['Stationarity_kpss'])
    else:
        Stationarity_kpss = pd.Series(sm.tsa.stattools.kpss(Y_Data[Target_name])[0:3],
                                      index=['Test Statistics', 'p-value', 'Used Lag'])
        for key, value in sm.tsa.stattools.kpss(Y_Data[Target_name])[3].items():
            Stationarity_kpss['Critical Value(%s)'%key] = value
            Stationarity_kpss = pd.DataFrame(Stationarity_kpss, columns=['Stationarity_kpss'])
    return Stationarity_kpss


# 정상성 테스트
def stationarity_ADF_KPSS(Residual):
    # ADF
    # Null Hypothesis: The Time-series is non-stationalry
    Stationarity = pd.Series(sm.tsa.stattools.adfuller(Residual)[0:3], 
                             index=['Test Statistics', 'p-value', 'Used Lag'])
    for key, value in sm.tsa.stattools.adfuller(Residual)[4].items():
        Stationarity['Critical Value(%s)'%key] = value
    Stationarity_ADF = pd.DataFrame(Stationarity, columns=['Stationarity_ADF'])
    
    # KPSS
    # Null Hypothesis: The Time-series is stationalry
    Stationarity = pd.Series(sm.tsa.stattools.kpss(Residual)[0:3], 
                             index=['Test Statistics', 'p-value', 'Used Lag'])
    for key, value in sm.tsa.stattools.kpss(Residual)[3].items():
        if key != '2.5%':
            Stationarity['Critical Value(%s)'%key] = value
    Stationarity_KPSS = pd.DataFrame(Stationarity, columns=['Stationarity_KPSS'])
    
    # 정리
    Stationarity = pd.concat([Stationarity_ADF, Stationarity_KPSS], 
                             join='outer', axis=1)
    
    return Stationarity