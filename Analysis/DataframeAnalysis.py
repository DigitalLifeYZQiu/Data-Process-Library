import os
import numpy as np
import pyarrow.parquet as pq
from arch.unitroot import ADF, PhillipsPerron, DFGLS, KPSS, ZivotAndrews, VarianceRatio
from scipy.fftpack import fft, fftfreq
import pandas as pd


class DataframeAnalysis():
    def __init__(self, root_path, data_path):


        # Load data to project. Currently supporting '.csv', '.xlsx', and '.parquet'.
        # :param root_path: The directory of all data
        # :param data_path: The path of a specific dataset

        self.root_path = root_path
        self.data_path = data_path
        if data_path.endswith('.csv'):
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            self.df_raw = df_raw
        elif data_path.endswith('.xlsx'):
            df_raw = pd.read_excel(os.path.join(self.root_path, self.data_path))
            self.df_raw = df_raw
        elif data_path.endswith('.parquet'):
            parquet_file = pq.ParquetFile(os.path.join(self.root_path, self.data_path))
            df_raw = parquet_file.read().to_pandas()
            self.df_raw = df_raw

    # * 统计量
    def getShape(self):
        """
        Get the shape of target dataset.
        :return: The shape of target dataset.
        """
        # 获取数据形状：（序列长度，变量数）
        return self.df_raw.shape

    def getAverageColumn(self, start_col=None, end_col=None):
        """
        本地算法快速训练接口。

        仅需提供本地资源和训练相关的信息，
        即可在Anylearn后端引擎启动自定义算法/数据集的训练：

        - 算法路径（文件目录或压缩包）
        - 数据集路径（文件目录或压缩包）
        - 训练启动命令
        - 训练输出路径
        - 训练超参数

        本接口封装了Anylearn从零启动训练的一系列流程：

        - 算法注册、上传
        - 数据集注册、上传
        - 训练项目创建
        - 训练任务创建

        本地资源初次在Anylearn注册和上传时，
        会在本地记录资源的校验信息。
        下一次调用快速训练或快速验证接口时，
        如果提供了相同的资源信息，
        则不再重复注册和上传资源，
        自动复用远程资源。

        如有需要，也可向本接口传入已在Anylearn远程注册的算法或数据集的ID，
        省略资源创建的过程。

        Parameters
        ----------
        algorithm_id : :obj:`str`, optional
            已在Anylearn远程注册的算法ID。
        algorithm_cloud_name: :obj:`str`, optional
            指定的算法在Anylearn云环境中的名称。
            同一用户的自定义算法的名称不可重复。
            如有重复，则复用已存在的同名算法，
            算法文件将被覆盖并提升版本。
            原有版本仍可追溯。
        algorithm_local_dir : :obj:`str`, optional
            本地算法目录路径。
        algorithm_git_ref : :obj:`str`, optional
            算法Gitea代码仓库的版本号（可以是commit号、分支名、tag名）。
            使用本地算法时，如未提供此参数，则取本地算法当前分支名。
        algorithm_force_update : :obj:`bool`, optional
            在同步算法的过程中是否强制更新算法，如为True，Anylearn会对未提交的本地代码变更进行自动提交。默认为False。
        algorithm_entrypoint : :obj:`str`, optional
            启动训练的入口命令。
        algorithm_output : :obj:`str`, optional
            训练输出模型的相对路径（相对于算法目录）。
        algorithm_hyperparams : :obj:`dict`, optional
            训练超参数字典。
            超参数将作为训练启动命令的参数传入算法。
            超参数字典中的键应为长参数名，如 :obj:`--param` ，并省略 :obj:`--` 部分传入。
            如需要标识类参数（flag），可将参数的值设为空字符串，如 :obj:`{'my-flag': ''}` ，等价于 :obj:`--my-flag` 传入训练命令。
            默认为空字典。
        algorithm_hyperparams_prefix : :obj:`str`, optional
            训练超参数键前标识，可支持hydra特殊命令行传参格式的诸如 :obj:`+key1` 、 :obj:`++key2` 、 空前置 :obj:`key3` 等需求，
            默认为 :obj:`--` 。
        algorithm_hyperparams_delimeter :obj:`str`, optional
            训练超参数键值间的分隔符，默认为空格 :obj:` ` 。
        algorithm_envs : :obj:`dict`, optional
            训练环境变量字典。
        dataset_hyperparam_name : :obj:`str`, optional
            启动训练时，数据集路径作为启动命令参数传入算法的参数名。
            需指定长参数名，如 :obj:`--data` ，并省略 :obj:`--` 部分传入。
            数据集路径由Anylearn后端引擎管理。
            默认为 :obj:`dataset` 。
        dataset_id : :obj:`str`, optional
            已在Anylearn远程注册的数据集ID。
        dataset_cloud_names : :obj:`List[str]`, optional
            训练任务需使用的Anylearn云环境中的数据集的名称
        model_hyperparam_name : :obj:`str`, optional
            启动训练时，模型路径作为启动命令参数传入算法的参数名。
            需指定长参数名，如 :obj:`--model` ，并省略 :obj:`--` 部分传入。
            模型路径由Anylearn后端引擎管理。
            默认为 :obj:`model` 。
        model_id : :obj:`str`, optional
            已在Anylearn远程注册/转存的模型ID。
        model_cloud_names : :obj:`List[str]`, optional
            训练任务需使用的Anylearn云环境中的模型的名称
        pretrain_hyperparam_name: :obj:`str`, optional
            启动训练时，前置训练结果（间接抽象为“预训练”，即"pretrain"）路径作为启动命令参数传入算法的参数名。
            需指定长参数名，如 :obj:`--pretrain` ，并省略 :obj:`--` 部分传入。
            预训练结果路径由Anylearn后端引擎管理。
            默认为 :obj:`pretrain` 。
        pretrain_task_id: :obj:`List[str]` | :obj:`str`, optional
            在Anylearn进行过的训练的ID，一般为前缀TRAI的32位字符串。
            Anylearn会对指定的训练进行结果抽取并挂载到新一次的训练中。
        project_id : :obj:`str`, optional
            已在Anylearn远程创建的训练项目ID。
        tags : :obj:`list`, optional
            训练任务标签列表
        task_name : :obj:`str`, optional
            训练任务名称。
            若值为非空，则由SDK自动生成8位随机字符串作为训练任务名称。
        task_description : :obj:`str`, optional
            训练任务详细描述。
            若值为非空，
            且参数 :obj:`algorithm_force_update` 为 :obj:`True` 时，
            则Anylearn在自动提交本地算法变更时，
            会将此值作为commit message同步至远端
        image_name : :obj:`str`, optional
            训练使用的Anylearn云环境中的镜像的名称。
        quota_group_request : :obj:`dict`, optional
            训练所需计算资源组中资源数量。
            需指定 :obj:`key` 为 :obj:`name` , :obj:`value` 为 :obj:`资源组名称` ，若未指定则使用Anylearn后端的 :obj:`default` 资源组中的默认资源套餐。
        num_nodes : :obj:`int`, optional
            分布式训练需要的节点数。
        nproc_per_node : :obj:`int`, optional
            分布式训练每个节点运行的进程数。

        Returns
        -------
        TrainTask
            创建的训练任务对象
        Algorithm
            在快速训练过程中创建或获取的算法对象
        Dataset
            在快速训练过程中创建或获取的数据集对象
        Project
            创建的训练项目对象
        """

        # Get the average of each column in the target dataset from the starting column to the ending column.
        # :param start_col: The starting column.
        # :param end_col: The ending column.
        # :return: The average value of each column.

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

    def getVarianceColumn(self, start_col=None, end_col=None):
        """
        Get the variance of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The variance value of each column.
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
        '''
        Get the standard deviation of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The standard deviation value of each column.
        '''
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
        '''
        Get the median of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: THe ending column.
        :return: The median of each column.
        '''
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
        '''
        Get the quantile of each column in the target dataset from the starting column to the ending column.
        :param percent: The percentile of each column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The quantile of each column.
        '''
        # 获取数据每一列的分位数：定义percent值以设置分为数
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        quantile = df.quantile(percent, axis=0)
        return quantile

    def getMaxColumn(self, start_col=None, end_col=None):
        '''
        Get the max of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The max of each column.
        '''
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
        '''
        Get the min of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The min of each column.
        '''
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
        '''
        Get the cross correlation of each column in the target dataset from the starting column to the ending column.
        :param method: The calculation method of cross correlation ('pearson' | 'kendall' | 'spearman')
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The cross correlation of each column.
        '''
        # 获取所有序列两两之间的互相关性：定义method以指定计算相关性标准（'pearson' | 'kendall' | 'spearman'）
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:, start_col:end_col]
        return df.corr(method)

    def getSelfCorr(self, lag=1, start_col=None, end_col=None):
        '''
        Get the self correlation of each column in the target dataset from the starting column to the ending column.
        :param lag: The lagging length used to calculate self correlation.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The self correlation of each column.
        '''
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
        '''
        Get the ADF of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The ADF result of each column.
        '''
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
        '''
        Get the phillips perron result of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The phillips perron result of each column.
        '''
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
        :param end_col: The ending column.
        :return: The DF-GLS result of each column.
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
        '''
        Get the KPSS result of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The KPSS result of each column.
        '''
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
        '''
        Get the Zivot-Andrews result of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The Zivot-Andrews result of each column.
        '''
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
        '''
        Get the Variance Ratio result of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The Variance Ratio result of each column.
        '''
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
        '''
        Get the Fast Fourier Transform result of a certain column in the target dataset.
        :param col: The input column.
        :param top_k_seasons: The number of top k seasons.
        :return: The Fast Fourier Transform result of a certain column in the target dataset.
        '''
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
        '''
        Get the index containing Nan in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :return: The index containing Nan in the target dataset from the starting column to the ending column.
        '''
        # 获得包含缺失值的index条目
        if start_col == None:
            start_col = self.df_raw.columns[0]
        if end_col == None:
            end_col = self.df_raw.columns[-1]
        data_nan_time = self.df_raw[self.df_raw.loc[:, start_col:end_col].isnull().values == True].index.unique()
        return data_nan_time

    def getInterpolate(self, start_col=None, end_col=None, **kwargs):
        '''
        Get the interpolate result of each column in the target dataset from the starting column to the ending column.
        :param start_col: The starting column.
        :param end_col: The ending column.
        :param kwargs: The arguments of interpolate.
        :return: The interpolate result of each column in the target dataset from the starting column to the ending column.
        '''
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

