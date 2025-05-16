import os
import numpy as np
import pyarrow.parquet as pq
from arch.unitroot import ADF, PhillipsPerron, DFGLS, KPSS, ZivotAndrews, VarianceRatio
from scipy.fftpack import fft, fftfreq
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
# from typing import Dict, Tuple


class DataframeAnalysis():
    def __init__(self, root_path : Optional[str]=None, data_path : Optional[str]=None, dataFrame : Optional[pd.DataFrame]=None, skip: int = 0) -> None:
        # Load data to project. Currently supporting '.csv', '.xlsx', and '.parquet'.
        # :param root_path: The directory of all data
        # :param data_path: The path of a specific dataset
        
        if root_path is not None and data_path is not None:
            self.root_path = root_path
            self.data_path = data_path
            print(f"DataAnalysis loading data from: {root_path}/{data_path}")
            if data_path.endswith('.csv'):
                df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path),skiprows=skip)
                self.df_raw = df_raw
            elif data_path.endswith('.xlsx'):
                df_raw = pd.read_excel(os.path.join(self.root_path, self.data_path))
                self.df_raw = df_raw
            elif data_path.endswith('.parquet'):
                parquet_file = pq.ParquetFile(os.path.join(self.root_path, self.data_path))
                df_raw = parquet_file.read().to_pandas()
                self.df_raw = df_raw
        if dataFrame is not None:
            print(f"DataAnalysis loading data from DataFrame with shape: {dataFrame.shape}")
            self.df_raw = dataFrame

    # * 统计量
    def getShape(self):
        """
        Get the shape of target dataset.

        :return: The shape of target dataset.
        """
        # 获取数据形状：（序列长度，变量数）
        return self.df_raw.shape

    def getAverageColumn(self, start_col : str = None, end_col : str = None):
        """
        Get the average of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The average value of each column.
        :rtype: `pd.DataFrame`
        """
        # 获取数据每一列的均值
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        average = df.mean(axis=0)
        average_df = pd.DataFrame()
        average_df['feature'] = average.index
        average_df['average'] = average.values
        return average_df
    
    def plot_column_plotly(self, columns : list = None, start_point : int = 0, length : int = 96):
        '''
        Descriptions:
            Plot function for target column using Plotly.
        Inputs:
            choice of target column -> string format
        Outputs:
            Plotly image format
        '''
        if columns == None:
            columns = [self.df_raw.columns[-1]]
        fig = go.Figure()
        for col in columns:
            df = self.df_raw.loc[start_point : start_point + length, col]
            df_choice = np.array(df)
            length = len(df_choice)
        
            fig.add_trace(go.Scatter(x=np.arange(length), y=df_choice, mode='lines', name=f"Feature: {col}"))
        fig.update_layout(title=f'Dataset:{self.root_path}/{self.data_path}', xaxis_title='Timepoint',
                          yaxis_title=f'Feature value: {columns}')
        return fig

    def getVarianceColumn(self, start_col=None, end_col=None):
        """
        Get the variance of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The variance value of each column.
        :rtype: `pd.DataFrame`
        """
        # 获取数据每一列的方差
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        var = df.var(axis=0)
        var_df = pd.DataFrame()
        var_df['feature'] = var.index
        var_df['variance'] = var.values
        return var_df

    def getStdColumn(self, start_col=None, end_col=None):
        """
        Get the standard deviation of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :return: The standard deviation value of each column.
        :rtype: `pd.DataFrame`
        """
        # 获取数据每一列的标准差
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        std = df.std(axis=0)
        std_df = pd.DataFrame()
        std_df['feature'] = std.index
        std_df['standard deviation'] = std.values
        return std_df

    def getMedianColumn(self, start_col=None, end_col=None):
        """
        Get the median of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The median of each column.
        :rtype: `pd.DataFrame`
        """
        # 获取数据每一列的中位数
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        median = df.median(axis=0)
        median_df = pd.DataFrame()
        median_df['feature'] = median.index
        median_df['median'] = median.values
        return median_df

    def getQuantileColumn(self, percent=[1 / 4, 2 / 4, 3 / 4], start_col=None, end_col=None):
        """
        Get the quantile of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :param end_col: The ending column.
        :returns: The quantile of each column.
        :rtype: `pd.DataFrame`
        """
        # 获取数据每一列的分位数：定义percent值以设置分为数
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        quantile = df.quantile(percent, axis=0)
        return quantile

    def getMaxColumn(self, start_col=None, end_col=None):
        """
        Get the max of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :return: The max of each column.
        :rtype: `pd.DataFrame`
        """
        # 获取数据每一列的最大值
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        maxval = df.max(axis=0)
        maxval_df = pd.DataFrame()
        maxval_df['feature'] = maxval.index
        maxval_df['max value'] = maxval.values
        return maxval_df

    def getMinColumn(self, start_col=None, end_col=None):
        """
        Get the min of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :return: The min of each column.
        :rtype: `pd.DataFrame`
        """
        # 获取数据每一列的最小值
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        minval = df.min(axis=0)
        minval_df = pd.DataFrame()
        minval_df['feature'] = minval.index
        minval_df['min value'] = minval.values
        return minval_df

    # * 相关性
    def getCorr(self, method='pearson', start_col=None, end_col=None):
        """
        Get the cross correlation of each column in the target dataset from the starting column to the ending column.

        :param method: The calculation method of cross correlation ('pearson' | 'kendall' | 'spearman')
        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The cross correlation of each column.
        :rtype: `pd.DataFrame`
        """
        # 获取所有序列两两之间的互相关性：定义method以指定计算相关性标准（'pearson' | 'kendall' | 'spearman'）
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        return df.corr(method)

    def getSelfCorr(self, lag=1, start_col=None, end_col=None):
        """
        Get the self correlation of each column in the target dataset from the starting column to the ending column.

        :param lag: The lagging length used to calculate self correlation.
        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The self correlation of each column.
        :rtype: `pd.DataFrame`
        """
        # 获取所有序列自相关系数：定义lag以指定计算自相关的滞后期数（时间间隔）
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        autocorr_lag_dict = {}
        for i in df.columns:
            ts = pd.Series(self.df_raw[i].values, index=self.df_raw.index)
            autocorr_lag = ts.autocorr(lag)
            autocorr_lag_dict[i] = autocorr_lag
        autocorr_df = pd.DataFrame(autocorr_lag_dict.items(), columns=['feature', 'self correlation'])
        return autocorr_df

    # * 平稳性
    def getADF(self, start_col=None, end_col=None):
        """
        Get the ADF of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The ADF result of each column.
        :rtype: `dict`
        """
        # 获取所有序列的ADF平稳性测试结果
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        ADFresult = {}
        for i in df.columns:
            result = ADF(self.df_raw[i].values)
            ADFresult[i] = {"Test Statistic": result.stat, "P-value": result.pvalue, "Lags": result.lags,
                            "Trend": result.trend, "Summary": result.summary()}
        return ADFresult

    def getPhillipsPerron(self, start_col=None, end_col=None):
        """
        Get the phillips perron result of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The phillips perron result of each column.
        :rtype: `dict`
        """
        # 获取所有序列的Phillips-Perron平稳性测试结果
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        PhillipsPerronresult = {}
        for i in df.columns:
            result = PhillipsPerron(self.df_raw[i].values)
            PhillipsPerronresult[i] = {"Test Statistic": result.stat, "P-value": result.pvalue, "Lags": result.lags,
                                       "Trend": result.trend, "Summary": result.summary()}
        return PhillipsPerronresult

    def getDFGLS(self, start_col=None, end_col=None):
        '''
        Get the DF-GLS result of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The DF-GLS result of each column.
        :rtype: `dict`
        '''
        # 获取所有序列的DF-GLS平稳性测试结果
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        DFGLSresult = {}
        for i in df.columns:
            result = DFGLS(self.df_raw[i].values)
            DFGLSresult[i] = {"Test Statistic": result.stat, "P-value": result.pvalue, "Lags": result.lags,
                              "Trend": result.trend, "Summary": result.summary()}
        return DFGLSresult

    def getKPSS(self, start_col=None, end_col=None):
        """
        Get the KPSS result of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The KPSS result of each column.
        :rtype: `dict`
        """
        # 获取所有序列的KPSS平稳性测试结果
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        KPSSresult = {}
        for i in df.columns:
            result = KPSS(self.df_raw[i].values)
            KPSSresult[i] = {"Test Statistic": result.stat, "P-value": result.pvalue, "Lags": result.lags,
                             "Trend": result.trend, "Summary": result.summary()}
        return KPSSresult

    def getZivotAndrews(self, start_col=None, end_col=None):
        """
        Get the Zivot-Andrews result of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The Zivot-Andrews result of each column.
        :rtype: `dict`
        """
        # 获取所有序列的Zivot-Andrew平稳性测试结果
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        ZivotAndrewsresult = {}
        for i in df.columns:
            result = ZivotAndrews(self.df_raw[i].values)
            ZivotAndrewsresult[i] = {"Test Statistic": result.stat, "P-value": result.pvalue, "Lags": result.lags,
                                     "Trend": result.trend, "Summary": result.summary()}
        return ZivotAndrewsresult

    def getVarianceRatio(self, start_col=None, end_col=None):
        """
        Get the Variance Ratio result of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The Variance Ratio result of each column.
        :rtype: `dict`
        """
        # 获取所有序列的Variance Ratio平稳性测试结果
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        VarianceRatioresult = {}
        for i in df.columns:
            result = VarianceRatio(self.df_raw[i].values)
            VarianceRatioresult[i] = {"Test Statistic": result.stat, "P-value": result.pvalue, "Lags": result.lags,
                                      "Trend": result.trend, "Summary": result.summary()}
        return VarianceRatioresult

    # * 周期性分析
    def getFFTtopk(self, col, top_k_seasons=3):
        """
        Get the Fast Fourier Transform result of a certain column in the target dataset.

        :param col: The input column.
        :param top_k_seasons: The number of top k seasons.
        :returns: The Fast Fourier Transform result of a certain column in the target dataset.
        """
        # 获得k个最主要的周期
        if col not in self.df_raw.columns:
            print(f"column {col} not found")
            return None
        fft_series = fft(self.df_raw.loc[:, col].values)
        power = np.abs(fft_series)
        sample_freq = fftfreq(fft_series.size)
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        powers = power[pos_mask]
        top_k_ids = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
        top_k_power = powers[top_k_ids]
        fft_periods = (1 / freqs[top_k_ids]).astype(int)
        sample_freq = pd.DataFrame(sample_freq, columns=['fft results'])
        return {"top_k_power": top_k_power, "fft_periods": fft_periods}, sample_freq

    # * 缺失值分析
    def getNanIndex(self, start_col=None, end_col=None):
        """
        Get the index containing Nan in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :returns: The index containing Nan in the target dataset from the starting column to the ending column.
        :rtype: `np.array`
        """
        # 获得包含缺失值的index条目
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        data_nan_time = self.df_raw[self.df_raw.loc[:, start_col:end_col].isnull().values == True].index.unique()
        flag = False
        if data_nan_time.values.size != 0:
            flag=True
        return data_nan_time, flag

    def getInterpolate(self, start_col=None, end_col=None, **kwargs):
        """
        Get the interpolate result of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :param kwargs: The arguments of interpolate.
        :returns: The interpolate result of each column in the target dataset from the starting column to the ending column.
        """
        # 插值填补函数(通过**kwargs传入interpolate函数的参数)
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        NullNum = self.df_raw.loc[:, start_col:end_col].isnull().sum()
        if True in [i > 0 for i in NullNum]:
            print('kwargs:', kwargs)
            new_df = self.df_raw.loc[:, start_col:end_col].interpolate(**kwargs)
        else:
            new_df = self.df_raw.loc[:, start_col:end_col]
        self.df_raw.loc[:, start_col:end_col] = new_df
        return self.df_raw
    
    #* 离群点检测
    def checkOutlierIQR(self, start_col=None, end_col=None, **kwargs):
        """
        Check the existence of outlier in the target dataset from the starting column to the ending column.
        return true if outlier exists.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :param kwargs: The arguments of interpolate.
        :returns: The existence of outlier under IQR method in data.
        """
        # 异常值替换函数(通过**kwargs传入interpolate函数的参数)
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        new_df = self.df_raw.loc[:, start_col:end_col]
        flag = False
        outliers = []
        for i in new_df.columns:
            Q1 = new_df[i].quantile(0.25)
            Q3 = new_df[i].quantile(0.75)
            IQR = Q3 - Q1
            outlier_i = new_df[i][(new_df[i] < Q1 - 1.5 * IQR) | (new_df[i] > Q3 + 1.5 * IQR)]
            outliers.append(outlier_i)
            if outlier_i.values.size != 0:
                flag= True
        return outliers, flag

    def getOutlierIQR(self, start_col=None, end_col=None, **kwargs):
        """
        Get the outlier replace result of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :param kwargs: Other arguments.
        :returns: The outlier replacement of each column in the target dataset from the starting column to the ending column.
        """
        # 异常值替换函数(通过**kwargs传入interpolate函数的参数)
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        new_df = self.df_raw.loc[:, start_col:end_col]
        for i in new_df.columns:
            Q1 = new_df[i].quantile(0.25)
            Q3 = new_df[i].quantile(0.75)
            IQR = Q3 - Q1
            new_df[i] = new_df[i].mask((new_df[i] < Q1 - 1.5 * IQR), Q1 - 1.5 * IQR).mask((new_df[i] > Q3 + 1.5 * IQR), Q3 + 1.5 * IQR)
        self.df_raw.loc[:, start_col:end_col] = new_df
        return self.df_raw

    def checkDateContinuity(self, date_col, freq=None, **kwargs):
        """
        Check the date continuity of each column in the target dataset from the starting column to the ending column.
        return true when missing dates found.

        :param date_col: The column of timestamp.
        :type date_col: `str`
        :param kwargs: Other arguments
        :returns: The interpolate result of each column in the target dataset from the starting column to the ending column.
        """
        # 确保时间戳列为 datetime64[ns] 类型
        # if not pd.api.types.is_datetime64_any_dtype(self.df_raw[date_col]):
        #     self.df_raw[date_col] = pd.to_datetime(self.df_raw[date_col])
        # self.df_raw[date_col] = pd.to_datetime(self.df_raw[date_col], format='%Y%m%d%H%M')
        self.df_raw[date_col] = pd.to_datetime(self.df_raw[date_col])
            
        # 去重并排序时间戳
        timestamps = self.df_raw[date_col].drop_duplicates().sort_values()
        
        # 检查是否有足够的时间戳用于连续性判断
        if timestamps.size < 2:
            return pd.DatetimeIndex([]), True
        
        # 未给定时序频率时默认为 15 秒级采样
        if freq is None:
            freq='15min'
        
        # 生成理论上的freq 频率时间戳序列
        start_time = timestamps.min()
        end_time = timestamps.max()
        expected_range = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # 找出缺失的时间戳
        missing_timestamps = expected_range.difference(timestamps)
        return missing_timestamps, len(missing_timestamps)==0, timestamps, expected_range
    
    def getMovingAverage(self, start_col=None, end_col=None, window_size=3, **kwargs):
        """
        Get the moving average smoothing result of each column in the target dataset from the starting column to the ending column.

        :param start_col: The starting column.
        :type start_col: `str`
        :param end_col: The ending column.
        :type end_col: `str`
        :param window_size: The size of moving average smoothing.
        :type window_size: `int`
        :param kwargs: Other arguments.
        :returns: The moving average of each column in the target dataset from the starting column to the ending column.
        """
        # 异常值替换函数(通过**kwargs传入interpolate函数的参数)
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        new_df = self.df_raw.loc[:, start_col:end_col]
        for i in new_df.columns:
            new_df[i] = new_df[i].rolling(window=window_size).mean()
            new_df[i][0:window_size] = self.df_raw[i][0:window_size]
        self.df_raw.loc[:, start_col:end_col] = new_df
        return self.df_raw