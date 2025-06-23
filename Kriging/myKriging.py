from myPyKriging3D import MyOrdinaryKriging3D as OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D
import pykrige
import time
# from sklearn.svm import SVR
# from pykrige.rk import RegressionKriging
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression

# from sklearn.metrics import r2_score
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import numpy as np

'''
计划用torch改为GPU加速版本
'''
def training(df, 
             variogram_model="linear", 
             nlags = 8, 
             enable_plotting=False, 
             weight=False, 
             uk = False,
             cpu_on = False):
 
    coordinates = df[['x', 'y', 'z']].values
    target_values = df['target'].values


    # 普通kriging
    # enable_plotting 是画已知点方差随距离的变化情况
    # nlags 是距离分组数
    if uk:
        model = UniversalKriging3D(coordinates[:, 0].astype(float), coordinates[:, 1].astype(float), coordinates[:, 2].astype(float), target_values, variogram_model=variogram_model, weight=weight, nlags = nlags, enable_plotting=enable_plotting)
        return model
    # 因为不能训练都用gpu，要不然精确度比对不严谨，所以生成两个model
    if cpu_on == False:
        model = OrdinaryKriging3D(coordinates[:, 0].astype(float), 
                                coordinates[:, 1].astype(float), 
                                coordinates[:, 2].astype(float), 
                                target_values, 
                                variogram_model=variogram_model, 
                                weight=weight, 
                                nlags = nlags, 
                                enable_plotting=enable_plotting)
    else:
        model = pykrige.ok3d.OrdinaryKriging3D(coordinates[:, 0].astype(float), 
                                coordinates[:, 1].astype(float), 
                                coordinates[:, 2].astype(float), 
                                target_values, 
                                variogram_model=variogram_model, 
                                weight=weight, 
                                nlags = nlags, 
                                enable_plotting=enable_plotting)
    return model

def testing(df, 
            model,
            block_size = 10000, 
            cpu_on = False, 
            style = "gpu_b",
            multi_process = False,
            print_time = False,
            torch_ac = False, 
            compute_precision = True):
    coordinates_test = df[['x', 'y', 'z']].values
    
    num_zeros = (df['target'].values == 0).sum()
    print("输入的检验矩阵中有",num_zeros,"个0（gpukriging自身检验，外部程序不作考虑）")

    target_values_test = df['target'].values.astype(float)
    if style == "cpu_nb":
        predict_values_test, ss = model.execute('points', 
                                            coordinates_test[:, 0].astype(float), 
                                            coordinates_test[:, 1].astype(float), 
                                            coordinates_test[:, 2].astype(float)) 
    else:
        predict_values_test, ss = model.execute('points', 
                                            coordinates_test[:, 0].astype(float), 
                                            coordinates_test[:, 1].astype(float), 
                                            coordinates_test[:, 2].astype(float),
                                            block_size = block_size,
                                            cpu_on = cpu_on,
                                            multi_process = multi_process,
                                            print_time = print_time,
                                            torch_ac = torch_ac) 

    # 机器学习模型直接评估
    # print("模型的r2分数为",r2_score(target_values_test, predict_values_test))
    
    
    # 计算平均相对误差
    # print("模型的平均相对误差1为",np.abs(mean_absolute_error(target_values_test,predict_values_test)/np.mean(target_values_test)))
    if compute_precision == True:
    # 创造非0掩码，只计算真实值非0处的误差（真实值有0是因为模拟粒子数和划分数的比值还有循环数不够大）
        non_zero_mask = target_values_test != 0
        
        print("模型的平均相对误差2为",np.mean(np.abs(target_values_test[non_zero_mask]-predict_values_test[non_zero_mask])/target_values_test[non_zero_mask]))
    
    
    return predict_values_test,target_values_test

def testing_for_time(coordinates_test, 
                    model,
                    block_size = 10000, 
                    cpu_on = False, 
                    style = "gpu_b",
                    multi_process = False,
                    print_time = False,
                    torch_ac = False):
    
    begin = time.time()

    predict_values_test, ss = model.execute('points', 
                                        coordinates_test[:, 0].astype(float), 
                                        coordinates_test[:, 1].astype(float), 
                                        coordinates_test[:, 2].astype(float),
                                        block_size = block_size,
                                        cpu_on = cpu_on,
                                        multi_process = multi_process,
                                        print_time = print_time,
                                        torch_ac = torch_ac) 
    end = time.time()
    time_spend = end - begin
    return time_spend
            
'''
training()下的注释：
 # 对特征标准化
    # scaler = MinMaxScaler()
    # coordinates = scaler.fit_transform(coordinates)
    # SVR也不行
    # model = SVR(kernel='rbf', C=1.0, epsilon=0.1).fit(coordinates, target_values)
    
    # 直接用随机森林
    rf_model = RandomForestRegressor(n_estimators=1000)
    # rf_model.fit(coordinates, target_values)
    # model = rf_model
    
    # 线性回归
    # lr_model = LinearRegression(copy_X=False, fit_intercept=True)
    # lr_model.fit(coordinates, target_values)
    # model = lr_model
    #*********************************************************************************
    # 回归kriging(使用随机森林/线性回归)
    # model = RegressionKriging(method='ordinary3d', regression_model=rf_model, n_closest_points=50, weight=True)
    # model.fit(coordinates, coordinates, target_values)

testing()下的注释
    # 对特征标准化
    # coordinates_test = scaler.transform(coordinates_test)
    
    # predict_values_test = model.predict(coordinates_test, coordinates_test)
    # predict_values_test = model.predict(coordinates_test)
    
    # 普通kriging
    #**********************************************************************************
    # 回归kriging评估模型
    # print("Regression Score: ", model.regression_model.score(coordinates_test, target_values_test))
    # print("RK score: ", model.score(coordinates_test, coordinates_test, target_values_test))
    # print("模型的r2分数为",r2_score(target_values_test, predict_values_test))
    
    #**********************************************************************************
    
    
'''