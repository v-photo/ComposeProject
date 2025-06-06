import time
import numpy as np #numpy必须要放在torch前面
# 用torch进行execute()和exec_vector()的全局加速估计比cupy好不少，但暂时不写
import torch # torch.cdist() 可以稍微改善execute()里的耗时部分(分块大的时候)，但是不能与多线程兼容
import cupy as cp #由于内部会有使用pycuda加速的部分，所以不直接替换np
import scipy.linalg as spl
import gc #python内存回收
import multiprocessing as mp #多进程
from multiprocessing import Pool
from cupyx.scipy.spatial.distance import cdist as cp_cdist
from pykrige import core, variogram_models
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist, pdist, squareform
from pykrige.core import (
    P_INV,
    _krige,
    _adjust_for_anisotropy,
    _find_statistics,
    _initialize_variogram_model,
    _make_variogram_parameter_list,
    great_circle_distance,
    _variogram_residuals #这个的修改或许可以解决源附近预测
)

from pykrige.ok3d import OrdinaryKriging3D
from pykrige.compat_gstools import validate_gstools

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
# 使用代码之前还有一些额外操作


# 将协方差改为其它指标或许可以解决源附近预测（但太难了）

eps = 1.0e-10  # Cutoff for comparison to zero

# 暂时只重写pykrige.variogram_models 中的 linear_variogram_model 方法（对应于_krige中计算krige矩阵和右端向量部分）
def linear_variogram_model_gpu(m, d):
    """Linear model, m is [slope, nugget]"""
    # 使用cupy加速好，比自己写的快
    m = cp.asarray(m).astype(cp.float32)
    # d = cp.asarray(d).astype(cp.float32) 这里d传进来已经是cp.float32,这一步很耗时，可以注释
    slope = m[0]
    nugget = m[1]
    
    cp_value = slope * d + nugget
    return cp_value

# 重写pykrige.variogram_models 中的 exponential_variogram_model 方法（因为之前好像训练的指数模型）
def exponential_variogram_model_gpu(m, d):
    """Exponential model, m is [psill, range, nugget]"""
    # 使用cupy加速好，比自己写的快
    # begin_three = time.time() #定位计时部分开始
    m = cp.asarray(m).astype(cp.float32)
    # d = cp.asarray(d).astype(cp.float32)
    psill = m[0]
    range_ = m[1]
    nugget = m[2]
    
    cp_value = psill * (1.0 - cp.exp(-d / (range_ / 3.0))) + nugget
    # end_three = time.time() #定位计时部分结束
    # print("定位计时部分耗时", end_three - begin_three)
    return cp_value

# 重写pykrige.variogram_models 中的 power_variogram_model 方法（因为之前好像训练的指数模型）
def power_variogram_model_gpu(m, d):
    """Exponential model, m is [psill, range, nugget]"""
    # 使用cupy加速好，比自己写的快
    # begin_three = time.time() #定位计时部分开始
    m = cp.asarray(m).astype(cp.float32)
    # d = cp.asarray(d).astype(cp.float32)
    scale = m[0]
    exponent = m[1]
    nugget = m[2]
    
    cp_value = scale * d**exponent + nugget
    # end_three = time.time() #定位计时部分结束
    # print("定位计时部分耗时", end_three - begin_three)
    return cp_value

variogram_dict_gpu = {
    "linear": linear_variogram_model_gpu,
    "power": power_variogram_model_gpu,
    "gaussian": variogram_models.gaussian_variogram_model,
    "spherical": variogram_models.spherical_variogram_model,
    "exponential": exponential_variogram_model_gpu,
    "hole-effect": variogram_models.hole_effect_variogram_model,
}

# 定义计时修饰器
def runtime(name : str):
    def r_time(func):
        def wrap(*args, **kwargs):
            begin = time.time()
            temp = func(*args, **kwargs)
            end = time.time()
            print(name + "消耗时间为%.2f\n"%(end - begin))
            return temp
        return wrap
    return r_time

# 重写 pykrige.core 中的 _find_statistics 方法
def _find_statistics(
    X,
    y,
    variogram_function,
    variogram_model_parameters,
    coordinates_type,
    variogram_model, # 自己加的，用来传递__init__的variogram_model参数
    pseudo_inv=False,
):

    delta = np.zeros(y.shape)
    sigma = np.zeros(y.shape)

    for i in range(y.shape[0]):
        # skip the first value in the kriging problem
        if i == 0:
            continue

        else:
            begin = time.time()
            k, ss = _krige(
                X[:i, :],
                y[:i],
                X[i, :],
                variogram_model,
                variogram_function,
                variogram_model_parameters,
                coordinates_type,
                pseudo_inv,
            )
            
            end = time.time()
            print("第三-1-" + str(i) + "部分计时为",end - begin)
            
            # if the estimation error is zero, it's probably because
            # the evaluation point X[i, :] is really close to one of the
            # kriging system points in X[:i, :]...
            # in the case of zero estimation error, the results are not stored
            if np.absolute(ss) < eps:
                continue

            delta[i] = y[i] - k
            sigma[i] = np.sqrt(ss)
    
    begin = time.time()
    # only use non-zero entries in these arrays... sigma is used to pull out
    # non-zero entries in both cases because it is guaranteed to be positive,
    # whereas delta can be either positive or negative
    delta = delta[sigma > eps]
    sigma = sigma[sigma > eps]
    epsilon = delta / sigma

    end = time.time()
    print("第三-2部分计时为",end - begin)

    return delta, sigma, epsilon

# 打印cupy显存池占用
def print_mermory(string_1):
    mempool = cp.get_default_memory_pool()

    # 打印已用显存和总显存 (MB)
    print(string_1+f"内存池总显存: {mempool.total_bytes() / 1024 / 1024:.2f} MB")
    print(string_1+f"内存池已用显存: {mempool.used_bytes() / 1024 / 1024:.2f} MB")
    
# 重写 _krige 方法中计算距离的一部分函数为pycuda加速版本
kernel_code = """
// 计算距离的 GPU 一维核函数
__global__ void calc_distances1(float *d_X, float *d_coords, float *d_d, float *d_bd, int n_sample, int n_dim) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_sample) {
        // 计算样本点之间的距离
        for (int i = 0; i < n_sample; i++) {
            float dist = 0.0;
            // 下面循环时计算单个样本点与其他的距离
            for (int j = 0; j < n_dim; j++) {
                float diff = d_X[idx * n_dim + j] - d_X[i * n_dim + j];
                dist += diff * diff;
            }
            d_d[idx * n_sample + i] = sqrt(dist);
        }

        // 计算样本点与 coords 的距离
        float bd_dist = 0.0;
        for (int j = 0; j < n_dim; j++) {
            float diff = d_X[idx * n_dim + j] - d_coords[j];
            bd_dist += diff * diff;
        }
        d_bd[idx] = sqrt(bd_dist);
    }
}
// 计算距离的 GPU 二维核函数
__global__ void calc_distances2(float *d_X, float *d_coords, float *d_d, float *d_bd, int n_sample, int n_dim) {
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    // 修改为二维数组的努力 (block应该是二维，且block是n_sample*n_sample大小)
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    // int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    // int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    // int id = blockId * (blockDim.x * blockDim.y) + threadId;
    int id = idx_y * n_sample + idx_x;
    
    if (idx_x < n_sample && idx_y < n_sample) {
        // 计算样本点之间的距离
        float dist = 0.0;
        
        // 下面计算单个样本点与其他的距离
        for (int j = 0; j < n_dim; j++) {
            float diff = d_X[idx_x * n_dim + j] - d_X[idx_y * n_dim + j];
            dist += diff * diff;
        }
        
        d_d[id] = sqrt(dist);
        // 计算样本点与 coords 的距离（这里计算是正确的）
        float bd_dist = 0.0;
        for (int j = 0; j < n_dim; j++) {
            float diff = d_X[idx_x * n_dim + j] - d_coords[j];
            bd_dist += diff * diff;
        }
        d_bd[idx_x] = sqrt(bd_dist);
    }
}
"""
def calc_d_bd_gpu(X, coords, calc_distance_gpuDim : int):

    n_sample, n_dim = X.shape

    # 将 X 和 coords 的数据类型转换为 float32(无法优化)
    X = X.astype(np.float32)
    coords = coords.astype(np.float32)

    # 在 GPU 上分配内存(可使用gpuarray优化/cuda.In,cuda.Out,cuda.InOut优化)
    d_X = cuda.mem_alloc(X.nbytes)
    d_coords = cuda.mem_alloc(coords.nbytes)
    d_d = cuda.mem_alloc(n_sample * n_sample * np.dtype(np.float32).itemsize)
    d_bd = cuda.mem_alloc(n_sample * np.dtype(np.float32).itemsize)

    # 将数据传输到 GPU(可使用gpuarray优化/cuda.In,cuda.Out,cuda.InOut优化)
    cuda.memcpy_htod(d_X, X)
    cuda.memcpy_htod(d_coords, coords)

    # 编译 GPU 核函数(无法优化)
    mod = SourceModule(kernel_code)
    calc_distances1 = mod.get_function("calc_distances1")
    calc_distances2 = mod.get_function("calc_distances2")

    
    # 调用 GPU 核函数进行计算(cuda.In,cuda.Out,cuda.InOut优化)
    if calc_distance_gpuDim == 1:
        # 设置 GPU 核函数的网格和块大小(无法优化)
        block_size = 256
        grid_size = (n_sample*n_sample + block_size - 1) // block_size #保证计算的总数够填满距离矩阵
        calc_distances1(d_X, d_coords, d_d, d_bd, np.int32(n_sample), np.int32(n_dim), block=(block_size, 1, 1), grid=(grid_size, 1))
    elif calc_distance_gpuDim == 2:
        # 设置 GPU 核函数的网格和块大小(无法优化)
        block_size = (64, 16, 1)
        grid_size = ((n_sample + block_size[0] -1) // block_size[0], (n_sample + block_size[1] -1) // block_size[1]) #保证于脑内的构想线程模型矩阵尺寸相匹配
        calc_distances2(d_X, d_coords, d_d, d_bd, np.int32(n_sample), np.int32(n_dim), block=block_size, grid=grid_size)
        
    # 从 GPU 获取计算结果(可使用gpuarray优化)
    d = np.empty((n_sample , n_sample), dtype=np.float32)
    bd = np.empty(n_sample, dtype=np.float32)
    cuda.memcpy_dtoh(d, d_d)
    cuda.memcpy_dtoh(bd, d_bd)
    
    # 限制距离结果的精度为四位小数
    d = np.round(d, decimals=4)
    bd = np.round(bd, decimals=4)

    return d,bd
        
# 重写 pykrige.core 中的 _krige 方法（对应于_find_statistics中最耗时部分）
def _krige_has_print(
    X,
    y,
    coords,
    variogram_model, # 自己加的，用来传递__init__的variogram_model参数
    variogram_function,
    variogram_model_parameters,
    coordinates_type,
    pseudo_inv=False,
    # 重写 variogram_dict ，这是为了好重写pykrige.variogram_models 中的 linear_variogram_model 方法(我们暂时只重写linear_variogram_model)
    
):
    """这段代码实现了一个普通克里金插值的函数 _krige，用于计算给定坐标点处的克里金估计和估计误差的平方

    Parameters
    ----------
    X：一个二维数组，形状为 [n_samples, n_dim]，表示输入坐标数组，其中 n_samples 是样本数，n_dim 是坐标维度。
    
    y：一个一维数组，形状为 [n_samples]，表示测量值数组，与 X 中的坐标对应。
    
    coords：一个一维数组，形状为 [1, n_dim]，表示要评估克里金系统的点的坐标。
    
    variogram_function：一个可调用函数，用于评估变异函数模型。
    
    variogram_model_parameters：一个列表，包含用户指定的变异函数模型参数。
    
    coordinates_type：一个字符串，表示 X 数组中坐标的类型，可以是 "euclidean"（欧氏坐标）或 "geographic"（地理坐标）。
    
    pseudo_inv：一个布尔值，可选参数，默认为 False。指示是否使用伪逆矩阵求解克里金系统。如果为 True，则可以提高数值稳定性并平均冗余点，但可能需要更长的计算时间。

    Returns
    -------
    zinterp: float
        kriging estimate at the specified point
    sigmasq: float
        mean square error of the kriging estimate
    """

    zero_index = None
    zero_value = False

    # 计算除克里金点以外点的坐标数组之间的成对距离，并转换为方阵形式
    # 计算除克里金点以外点和克里金点之间的距离向量
    # 使用pycuda加速版本，目前只加速`euclidean`类型
    # 必须要限制小数位数，要不然GPU算不出来
    X = np.round(X, 2)
    coords = np.round(coords, 2)
    begin = time.time()
    if coordinates_type == "euclidean":
        # d_cpu = squareform(pdist(X, metric="euclidean"))
        # bd_cpu = np.squeeze(cdist(X, coords[None, :], metric="euclidean"))
        d, bd = calc_d_bd_gpu(X, coords[None, :], 2)

    # geographic coordinate distances still calculated in the old way...
    # assume X[:, 0] ('x') => lon, X[:, 1] ('y') => lat
    # also assume problem is 2D; check done earlier in initializing variogram
    elif coordinates_type == "geographic":
        x1, x2 = np.meshgrid(X[:, 0], X[:, 0], sparse=True)
        y1, y2 = np.meshgrid(X[:, 1], X[:, 1], sparse=True)
        d = great_circle_distance(x1, y1, x2, y2)
        bd = great_circle_distance(
            X[:, 0],
            X[:, 1],
            coords[0] * np.ones(X.shape[0]),
            coords[1] * np.ones(X.shape[0]),
        )

    # this check is done when initializing variogram, but kept here anyways...
    else:
        raise ValueError(
            "Specified coordinate type '%s' is not supported." % coordinates_type
        )

    end = time.time()
    print("计算平方距离矩阵的时间为",end - begin)
    
    begin = time.time()
    # check if kriging point overlaps with measurement point
    if np.any(np.absolute(bd) <= 1e-10):
        zero_value = True
        zero_index = np.where(bd <= 1e-10)[0][0]

    # 计算克里金矩阵和RHS（使用cupy加速了variogram_function（实际定位就是到下面的exponential_variogram_model））
    # 在内部把d和bd转换为cupy对象，不能直接改variogram_function，不止这里会用这函数

    # print(d,'\n',bd)

    n = X.shape[0]
    a = np.zeros((n + 1, n + 1), dtype = np.float32)

    begin = time.time()

    if n >= min(n//2, 800): #和下面的判断n应该同步，要不然会报错
        variogram_function = variogram_dict_gpu[variogram_model]
    a[:n, :n] = -np.array(variogram_function(variogram_model_parameters, d))

    print("d 的大小为：", d.size)
    end = time.time()
    print("设置krige矩阵中求解模型的时间为", end - begin)
    
    begin = time.time()
    
    np.fill_diagonal(a ,0)
    
    # 注意：不能在这之前直接把a变成cupy格式，要不然赋值时会报错
    a[n, :] = 1
    a[:, n] = 1
    a[n, n] = 0
    
    end = time.time()
    print("设置krige矩阵的其他时间为",end - begin)
    begin = time.time()
    
    # set up RHS（右端向量）
    b = np.zeros((n + 1, 1), dtype=np.float32)

    b[:n, 0] = -np.array(variogram_function(variogram_model_parameters, bd))

    if zero_value:
        b[zero_index, 0] = 0.0
    b[n, 0] = 1.0

    end = time.time()
    print("设置RHS（右端向量）的时间为", end - begin)
    begin = time.time()
    
    # solve（使用cupy加速，其会自动把传入的np格式转换为cp格式）
    # float32很重要，减少内存占用，精度下降不了多少

    if n >= min(n//2, 800):
        a = cp.round(cp.asarray(a), 1)
        b = cp.round(cp.asarray(b), 1)
        y = cp.round(cp.array(y, dtype = (cp.float32)), 1) # 把y转为cupy格式

        if pseudo_inv:
            res = cp.linalg.lstsq(a, b, rcond=None)[0]
        else:
            res = cp.linalg.solve(a, b)

        zinterp = cp.sum(res[:n, 0] * y)
        sigmasq = cp.sum(res[:, 0] * -b[:, 0])
    else:
        a = np.round(a, 1)
        b = np.round(b, 1)
        y = np.round(y.astype(np.float32), 1)
        if pseudo_inv:
            res = np.linalg.lstsq(a, b, rcond=None)[0]
        else:
            res = np.linalg.solve(a, b)

        zinterp = np.sum(res[:n, 0] * y)
        sigmasq = np.sum(res[:, 0] * -b[:, 0])

    # 用于比对的cpu程序
    # if pseudo_inv:
    #     res = np.linalg.lstsq(a, b, rcond=None)[0]
    # else:
    #     res = np.linalg.solve(a, b)

    # zinterp = np.sum(res[:n, 0] * y)
    # sigmasq = np.sum(res[:, 0] * -b[:, 0])

    end = time.time()
    print("求解的时间为",end - begin)
    
    # 释放内存，不然会爆
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    
    return zinterp, sigmasq

def _krige(
    X,
    y,
    coords,
    variogram_model, # 自己加的，用来传递__init__的variogram_model参数
    variogram_function,
    variogram_model_parameters,
    coordinates_type,
    pseudo_inv=False,
    
):

    zero_index = None
    zero_value = False

    X = np.round(X, 2)
    coords = np.round(coords, 2)
    begin = time.time()
    if coordinates_type == "euclidean":

        d, bd = calc_d_bd_gpu(X, coords[None, :], 2)

    elif coordinates_type == "geographic":
        x1, x2 = np.meshgrid(X[:, 0], X[:, 0], sparse=True)
        y1, y2 = np.meshgrid(X[:, 1], X[:, 1], sparse=True)
        d = great_circle_distance(x1, y1, x2, y2)
        bd = great_circle_distance(
            X[:, 0],
            X[:, 1],
            coords[0] * np.ones(X.shape[0]),
            coords[1] * np.ones(X.shape[0]),
        )

    # this check is done when initializing variogram, but kept here anyways...
    else:
        raise ValueError(
            "Specified coordinate type '%s' is not supported." % coordinates_type
        )

    end = time.time()
    
    begin = time.time()
    # check if kriging point overlaps with measurement point
    if np.any(np.absolute(bd) <= 1e-10):
        zero_value = True
        zero_index = np.where(bd <= 1e-10)[0][0]

    n = X.shape[0]
    a = np.zeros((n + 1, n + 1), dtype = np.float32)

    begin = time.time()

    if n >= min(n//2, 800): #和下面的判断n应该同步，要不然会报错
        variogram_function = variogram_dict_gpu[variogram_model]
    a[:n, :n] = -np.array(variogram_function(variogram_model_parameters, d))

    end = time.time()
    
    begin = time.time()
    
    np.fill_diagonal(a ,0)
    
    # 注意：不能在这之前直接把a变成cupy格式，要不然赋值时会报错
    a[n, :] = 1
    a[:, n] = 1
    a[n, n] = 0
    
    end = time.time()
    begin = time.time()
    
    # set up RHS（右端向量）
    b = np.zeros((n + 1, 1), dtype=np.float32)

    b[:n, 0] = -np.array(variogram_function(variogram_model_parameters, bd))

    if zero_value:
        b[zero_index, 0] = 0.0
    b[n, 0] = 1.0

    end = time.time()
    begin = time.time()

    if n >= min(n//2, 800):
        a = cp.round(cp.asarray(a), 1)
        b = cp.round(cp.asarray(b), 1)
        y = cp.round(cp.array(y, dtype = (cp.float32)), 1) # 把y转为cupy格式

        if pseudo_inv:
            res = cp.linalg.lstsq(a, b, rcond=None)[0]
        else:
            res = cp.linalg.solve(a, b)

        zinterp = cp.sum(res[:n, 0] * y)
        sigmasq = cp.sum(res[:, 0] * -b[:, 0])
    else:
        a = np.round(a, 1)
        b = np.round(b, 1)
        y = np.round(y.astype(np.float32), 1)
        if pseudo_inv:
            res = np.linalg.lstsq(a, b, rcond=None)[0]
        else:
            res = np.linalg.solve(a, b)

        zinterp = np.sum(res[:n, 0] * y)
        sigmasq = np.sum(res[:, 0] * -b[:, 0])

    end = time.time()
    
    # 释放内存，不然会爆
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    
    return zinterp, sigmasq  
# 继承pykrige.OrdinaryKriging3D，以测试其运行状况
class MyOrdinaryKriging3D(OrdinaryKriging3D):    
    @runtime(name = "init")
    def __init__(
        self,
        x,
        y,
        z,
        val,
        variogram_model="linear",
        variogram_parameters=None,
        variogram_function=None,
        nlags=6,
        weight=False,
        anisotropy_scaling_y=1.0,
        anisotropy_scaling_z=1.0,
        anisotropy_angle_x=0.0,
        anisotropy_angle_y=0.0,
        anisotropy_angle_z=0.0,
        verbose=False,
        enable_plotting=False,
        exact_values=True,
        pseudo_inv=False,
        pseudo_inv_type="pinv"
    ):  
        # 配置求广义逆的方式
        self.pseudo_inv = bool(pseudo_inv)
        self.pseudo_inv_type = str(pseudo_inv_type)
        if self.pseudo_inv_type not in P_INV:
            raise ValueError("pseudo inv type not valid: " + str(pseudo_inv_type))

        # 设置方差模型和参数
        self.variogram_model = variogram_model
        self.model = None

        # 查看exact_values是否是Bool值
        if not isinstance(exact_values, bool):
            raise ValueError("exact_values has to be boolean True or False")
        self.exact_values = exact_values


        # check if a GSTools covariance model is given
        # begin = time.time()
        if hasattr(self.variogram_model, "pykrige_kwargs"):
            # save the model in the class
            self.model = self.variogram_model
            validate_gstools(self.model)
            if self.model.field_dim < 3:
                raise ValueError("GSTools: model dim is not 3")
            self.variogram_model = "custom"
            variogram_function = self.model.pykrige_vario
            variogram_parameters = []
            anisotropy_scaling_y = self.model.pykrige_anis_y
            anisotropy_scaling_z = self.model.pykrige_anis_z
            anisotropy_angle_x = self.model.pykrige_angle_x
            anisotropy_angle_y = self.model.pykrige_angle_y
            anisotropy_angle_z = self.model.pykrige_angle_z
        if (
            self.variogram_model not in self.variogram_dict.keys()
            and self.variogram_model != "custom"
        ):
            raise ValueError(
                "Specified variogram model '%s' is not supported." % variogram_model
            )
        elif self.variogram_model == "custom":
            if variogram_function is None or not callable(variogram_function):
                raise ValueError(
                    "Must specify callable function for custom variogram model."
                )
            else:
                self.variogram_function = variogram_function
        else:
            # variogram_function 在这里被定义，然后到_find_statistics中，然后到_krige中
            # 只把传递给_find_statistics的variogram_dict改为 variogram_dict_gpu
            self.variogram_function = self.variogram_dict[self.variogram_model]
            
        # 检验x,y,z是否都为一维数据，防止出现错误
        self.X_ORIG = np.atleast_1d(
            np.squeeze(np.array(x, copy=True, dtype=np.float64))
        )
        self.Y_ORIG = np.atleast_1d(
            np.squeeze(np.array(y, copy=True, dtype=np.float64))
        )
        self.Z_ORIG = np.atleast_1d(
            np.squeeze(np.array(z, copy=True, dtype=np.float64))
        )
        self.VALUES = np.atleast_1d(
            np.squeeze(np.array(val, copy=True, dtype=np.float64))
        )

        self.verbose = verbose
        self.enable_plotting = enable_plotting
        if self.enable_plotting and self.verbose:
            print("Plotting Enabled\n")

        self.XCENTER = (np.amax(self.X_ORIG) + np.amin(self.X_ORIG)) / 2.0
        self.YCENTER = (np.amax(self.Y_ORIG) + np.amin(self.Y_ORIG)) / 2.0
        self.ZCENTER = (np.amax(self.Z_ORIG) + np.amin(self.Z_ORIG)) / 2.0
        self.anisotropy_scaling_y = anisotropy_scaling_y
        self.anisotropy_scaling_z = anisotropy_scaling_z
        self.anisotropy_angle_x = anisotropy_angle_x
        self.anisotropy_angle_y = anisotropy_angle_y
        self.anisotropy_angle_z = anisotropy_angle_z
        if self.verbose:
            print("Adjusting data for anisotropy...")
        self.X_ADJUSTED, self.Y_ADJUSTED, self.Z_ADJUSTED = _adjust_for_anisotropy(
            np.vstack((self.X_ORIG, self.Y_ORIG, self.Z_ORIG)).T,
            [self.XCENTER, self.YCENTER, self.ZCENTER],
            [self.anisotropy_scaling_y, self.anisotropy_scaling_z],
            [self.anisotropy_angle_x, self.anisotropy_angle_y, self.anisotropy_angle_z],
        ).T

        if self.verbose:
            print("Initializing variogram model...")
        
        vp_temp = _make_variogram_parameter_list(
            self.variogram_model, variogram_parameters
        )
        
        # end = time.time()
        # print("第一部分计时为", end - begin)
        # begin = time.time()
        
        (
            self.lags,
            self.semivariance,
            self.variogram_model_parameters,
        ) = _initialize_variogram_model(
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED, self.Z_ADJUSTED)).T,
            self.VALUES,
            self.variogram_model,
            vp_temp,
            self.variogram_function,
            nlags,
            weight,
            "euclidean",
        )
        
        # end = time.time()
        # print("第二部分计时为", end - begin)
        # begin = time.time()
        if self.verbose:
            if self.variogram_model == "linear":
                print("Using '%s' Variogram Model" % "linear")
                print("Slope:", self.variogram_model_parameters[0])
                print("Nugget:", self.variogram_model_parameters[1], "\n")
            elif self.variogram_model == "power":
                print("Using '%s' Variogram Model" % "power")
                print("Scale:", self.variogram_model_parameters[0])
                print("Exponent:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], "\n")
            elif self.variogram_model == "custom":
                print("Using Custom Variogram Model")
            else:
                print("Using '%s' Variogram Model" % self.variogram_model)
                print("Partial Sill:", self.variogram_model_parameters[0])
                print(
                    "Full Sill:",
                    self.variogram_model_parameters[0]
                    + self.variogram_model_parameters[2],
                )
                print("Range:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], "\n")
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print("Calculating statistics on variogram model fit...")

            # 把self.variogram_model设置为_find_statistics的参数传递, 方便后续判断
            # 这里进行了修改, 改为了当self.verbose时才计算统计值, 要不然太耗时
            self.delta, self.sigma, self.epsilon = _find_statistics(
                np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED, self.Z_ADJUSTED)).T,
                self.VALUES,
                self.variogram_function,
                self.variogram_model_parameters,
                "euclidean",
                self.variogram_model,
                self.pseudo_inv,
            )
            
            # end = time.time()
            # print("第三部分计时为", end - begin)
            self.Q1 = core.calcQ1(self.epsilon)
            self.Q2 = core.calcQ2(self.epsilon)
            self.cR = core.calc_cR(self.Q2, self.sigma)
            if self.verbose:
                print("Q1 =", self.Q1)
                print("Q2 =", self.Q2)
                print("cR =", self.cR, "\n")
        
    @runtime(name = "display_variogram_model")
    def display_variogram_model(self):
        """Displays variogram model with the actual binned data."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.lags[:len(self.lags)*6//10], self.semivariance[:len(self.lags)*6//10], "r*")
        ax.plot(
            self.lags[:len(self.lags)*6//10],
            self.variogram_function(self.variogram_model_parameters, self.lags[:len(self.lags)*6//10]),
            "k-",
        )
        plt.show()
    
    #exectue函数组件
    def process_block(self, 
                      i, 
                      block_size, 
                      nx, 
                      zpts, 
                      ypts, 
                      xpts, 
                      xyz_data, 
                      n_closest_points, 
                      backend, 
                      cpu_on, 
                      a, 
                      mask, 
                      print_time = False,
                      torch_ac = False):
        start_time = time.time()  # 记录开始时间

        start = i
        end = min(i + block_size, nx)

        # 分块构建xyz_points
        xyz_points_block = np.concatenate(
            (
                zpts[start:end, np.newaxis],
                ypts[start:end, np.newaxis],
                xpts[start:end, np.newaxis]
            ),
            axis=1
        )
        # time_begin = time.time()
        # print("xyz_points_block的shape为:",xyz_points_block.shape,",xyz_data的shape为：",xyz_data.shape)
        if cpu_on == False and torch_ac == True:
            # 将NumPy数组转换为PyTorch张量
            xyz_points_block_tensor = torch.from_numpy(xyz_points_block).cuda()
            xyz_data_tensor = torch.from_numpy(xyz_data).cuda()

            # 使用torch.cdist()计算成对距离
            bd = torch.cdist(xyz_points_block_tensor, xyz_data_tensor).cupy().astype(cp.float32)
            torch.cuda.empty_cache()
        elif cpu_on == True:
            bd = cdist(xyz_points_block, xyz_data, "euclidean") #cdist和exec_vector耗时最多，各占大概execute的一半
        else: # 普通gpu版本用cupyx cdist加速, 得到的bd是cupy
            # begin_cpdist = time.time()
            bd = cp_cdist(xyz_points_block, xyz_data, "euclidean")
            # end_cpdist = time.time()
            # print("cp_cdist计算时间为", end_cpdist - begin_cpdist)
            # 这里有两步意义不明的操作，
            # 1是回收cupy内存池(如果不回收会在_exec_vector里
            # 用cp.linalg.inv()是程序死循环，不知道为啥)
            # 2是cp.ones((1,1)).get() (随便的gpu到cpu的复制)，如果不这样cp.linalg.inv()会报cuSolver错误
            # free_memory_time_begin = time.time()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks() 
            # free_memory_time_end = time.time()
            # print("cp_cdist中回收显存消耗时间为",free_memory_time_end - free_memory_time_begin)
            # begin_bd_np = time.time()
            cp.ones((1,1)).get()
            # end_bd_np = time.time()
            # print("cp_cdist中cupy转numpy消耗时间为",end_bd_np - begin_bd_np)
            # print_mermory("cp_dist结束时")
        # time_end = time.time()
        # print("execute()中cdist消耗时间为：",time_end - time_begin)
        if n_closest_points is not None:
            from scipy.spatial import cKDTree
            tree = cKDTree(xyz_data)
            bd, bd_idx = tree.query(xyz_points_block, k=n_closest_points, eps=0.0)
            if backend == "loop":
                kvalues_block, sigmasq_block = self._exec_loop_moving_window(a, bd, mask[start:end], bd_idx)
            else:
                raise ValueError(
                    "Specified backend '{}' not supported "
                    "for moving window.".format(backend)
                )
        else:
            if backend == "vectorized":
                if cpu_on == False:
                    kvalues_block, sigmasq_block = self._exec_vector(a, bd, mask[start:end])
                else:
                    kvalues_block, sigmasq_block = self._exec_vector_cpu(a, bd, mask[start:end])
            elif backend == "loop":
                kvalues_block, sigmasq_block = self._exec_loop(a, bd, mask[start:end])
            else:
                raise ValueError(
                    "Specified backend {} is not supported for "
                    "3D ordinary kriging.".format(backend)
                )

        # 这里不清只放在cdist这清（可能没用，初步判断耗时和大小有关）
        # 内存回收(i//3 == 0时回收减少耗时)
        del xyz_points_block
        del bd
        if n_closest_points is not None:
            del tree
            del bd_idx
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks() 

        end_time = time.time()  # 记录结束时间
        # print_mermory(f"Block {i // block_size}结束时")
        if print_time:
            print(f"Block {i // block_size} processing time: {end_time - start_time:.2f} seconds")  # 打印时间差
        return kvalues_block, sigmasq_block

    @runtime(name = "execute")
    def execute(
        self,
        style,
        xpoints,
        ypoints,
        zpoints,
        mask=None,
        backend="vectorized",
        n_closest_points=None,
        block_size = 10000, #GPU分块大小
        cpu_on = False, #是否使用cpu计算
        multi_process = False,
        print_time = True, 
        torch_ac = False #是否使用torch加速cdist, 要用这个功能，内核要安装torch，要不然会报错
    ):
        if self.verbose:
            print("Executing Ordinary Kriging...\n")

        if style != "grid" and style != "masked" and style != "points":
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        xpts = np.atleast_1d(np.squeeze(np.array(xpoints, copy=True)))
        ypts = np.atleast_1d(np.squeeze(np.array(ypoints, copy=True)))
        zpts = np.atleast_1d(np.squeeze(np.array(zpoints, copy=True)))
        n = self.X_ADJUSTED.shape[0]
        nx = xpts.size
        ny = ypts.size
        nz = zpts.size
        a = self._get_kriging_matrix(n)

        if style in ["grid", "masked"]:
            if style == "masked":
                if mask is None:
                    raise IOError(
                        "Must specify boolean masking array when style is 'masked'."
                    )
                if mask.ndim != 3:
                    raise ValueError("Mask is not three-dimensional.")
                if mask.shape[0] != nz or mask.shape[1] != ny or mask.shape[2] != nx:
                    if (
                        mask.shape[0] == nx
                        and mask.shape[2] == nz
                        and mask.shape[1] == ny
                    ):
                        mask = mask.swapaxes(0, 2)
                    else:
                        raise ValueError(
                            "Mask dimensions do not match specified grid dimensions."
                        )
                mask = mask.flatten()
            npt = nz * ny * nx
            grid_z, grid_y, grid_x = np.meshgrid(zpts, ypts, xpts, indexing="ij")
            xpts = grid_x.flatten()
            ypts = grid_y.flatten()
            zpts = grid_z.flatten()
        elif style == "points":
            if xpts.size != ypts.size and ypts.size != zpts.size:
                raise ValueError(
                    "xpoints, ypoints, and zpoints must have "
                    "same dimensions when treated as listing "
                    "discrete points."
                )
            npt = nx
        else:
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        xpts, ypts, zpts = _adjust_for_anisotropy(
            np.vstack((xpts, ypts, zpts)).T,
            [self.XCENTER, self.YCENTER, self.ZCENTER],
            [self.anisotropy_scaling_y, self.anisotropy_scaling_z],
            [self.anisotropy_angle_x, self.anisotropy_angle_y, self.anisotropy_angle_z],
        ).T

        if style != "masked":
            mask = np.zeros(npt, dtype="bool")

        xyz_data = np.concatenate(
            (
                self.Z_ADJUSTED[:, np.newaxis],
                self.Y_ADJUSTED[:, np.newaxis],
                self.X_ADJUSTED[:, np.newaxis],
            ),
            axis=1,
        )
        # 实际上数据分块要在这一步执行，但前面已经修改了exec_vector，就展示妥协吧
        # print("xyz_points的尺寸为：",xyz_points.shape)
        # print("xyz_data的尺寸为：",xyz_data.shape)
        # xyz_points就是我们要分块的对象(884736, 3)
        kvalues_list = []
        sigmasq_list = []
        block_size = block_size
        if multi_process:  # 使用多进程
            if cpu_on == True: #cpu和gpu多进程编程有区别
                with Pool() as pool:
                    results = pool.starmap(
                        self.process_block,
                        [(i, block_size, nx, zpts, ypts, xpts, xyz_data, n_closest_points, backend, cpu_on, a, mask, print_time, torch_ac) for i in range(0, nx, block_size)]
                    )
                for result in results:
                    kvalues_list.append(result[0])
                    sigmasq_list.append(result[1])
            else:
                ctx = mp.get_context("spawn")
                print("cpu数目是:",mp.cpu_count())
                with ctx.Pool() as pool:
                    results = pool.starmap(
                        self.process_block,
                        [(i, block_size, nx, zpts, ypts, xpts, xyz_data, n_closest_points, backend, cpu_on, a, mask, print_time, torch_ac) for i in range(0, nx, block_size)]
                    )
                for result in results:
                    kvalues_list.append(result[0])
                    sigmasq_list.append(result[1])
                del results
        else:  # 使用单进程
            for i in range(0, nx, block_size):
                kvalues_block, sigmasq_block = self.process_block(i, block_size, nx, zpts, ypts, xpts, xyz_data, n_closest_points, backend, cpu_on, a, mask, print_time = print_time, torch_ac = torch_ac)
                kvalues_list.append(kvalues_block)
                sigmasq_list.append(sigmasq_block)
                del kvalues_block
                del sigmasq_block

        # 合并结果
        if cpu_on:
            kvalues = np.concatenate(kvalues_list, axis=0)
            sigmasq = np.concatenate(sigmasq_list, axis=0)
        else:
            kvalues = cp.concatenate(kvalues_list, axis=0).get()
            sigmasq = cp.concatenate(sigmasq_list, axis=0).get()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks() 
        if style == "masked":
            kvalues = np.ma.array(kvalues, mask=mask)
            sigmasq = np.ma.array(sigmasq, mask=mask)
        if style in ["masked", "grid"]:
            kvalues = kvalues.reshape((nz, ny, nx))
            sigmasq = sigmasq.reshape((nz, ny, nx))
        return kvalues, sigmasq

    @runtime(name = "_get_kriging_matrix")
    def _get_kriging_matrix(self, *args, **kwargs):
        return super()._get_kriging_matrix(*args, **kwargs)
        
    @runtime(name = "_exec_loop_moving_window")
    def _exec_loop_moving_window(self, *args, **kwargs):
        return super()._exec_loop_moving_window(*args, **kwargs)
    
    def _exec_vector_unit(self, b_cp, n, npt, mask, a_inv_cp):
        if (mask).any():
                raise NotImplementedError("mask is not supported in GPU.")

        
        x_cp = cp.dot(a_inv_cp, b_cp.reshape((npt, n + 1)).T).reshape((1, n + 1, npt)).T
    
        values_cp = cp.asarray(self.VALUES, dtype = cp.float32)
        # 这一步按值进行了修改（可对照原代码，主要是降低值的影响，结果没用）
        # kvalues_cp = cp.sum(x_cp[:, :n, 0] * values_cp * (1-(values_cp/cp.sum(values_cp))), axis=1)
        kvalues_cp = cp.sum(x_cp[:, :n, 0] * values_cp, axis=1)
        # kvalues_cp = cp.sum(x_cp[:, :n, 0] / values_cp ** 1/2, axis=1) * cp.mean(values_cp) ** 1/2
        # kvalues = cp.asnumpy(kvalues_cp)
        
        sigmasq_cp = cp.sum(x_cp[:, :, 0] * -b_cp[:, :, 0], axis=1)
        # sigmasq = cp.asnumpy(sigmasq_cp)
        # print_mermory("_exec_vector_unit结束时")
        return kvalues_cp, sigmasq_cp
        
    # @runtime(name = "_exec_vector")
    def _exec_vector(self, a, bd, mask):
        """Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets."""
        
        npt = bd.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zero_index = None
        zero_value = False

        a_cp = cp.asarray(a, dtype=cp.float32)
        # print_mermory("成功分配a_cp时") 
        # use the desired method to invert the kriging matrix
        if self.pseudo_inv:
            a_inv_cp = P_INV[self.pseudo_inv_type](a_cp)
        else:
            a_inv_cp = cp.linalg.inv(a_cp)
        
        if self.variogram_model not in variogram_dict_gpu.keys():
            raise NotImplementedError(f"variogram_model={self.variogram_model} is not supported in GPU.")
        else:
            variogram_function_cp = variogram_dict_gpu[self.variogram_model]
        
        
        # 由于我的bd一般比self.eps大，这一步又很消耗时间，所以默认先注释掉
        # if np.any(np.absolute(bd) <= self.eps):
        #     zero_value = True
        #     zero_index = np.where(np.absolute(bd) <= self.eps)
        b = cp.ones((npt, n + 1, 1), dtype=cp.float32)
        b[:, :n, 0] = -variogram_function_cp(self.variogram_model_parameters, bd)

        if zero_value and self.exact_values:
            b[zero_index[0], zero_index[1], 0] = 0.0
        
        b[:, n, 0] = 1.0

        # begin = time.time()
        kvalues, sigmasq = self._exec_vector_unit(b, n, npt, mask, a_inv_cp)
        # end = time.time()
        # print("_exec_vector_unit消耗的时间为:",end - begin)
        
        # begin = time.time()
        # 注释掉后面回收耗时就会增加
        del a_cp,a_inv_cp,b
        # print_mermory("_exec_vector结束未free时时")    
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()  
        # print_mermory("_exec_vector结束时")   
        # end = time.time()
        # print("_exec_vector回收显存消耗时间为", end - begin)  block_size增大，所有回收的总耗时也会增大，会增大很多倍
        return kvalues, sigmasq
        
    @runtime(name = "_exec_vector_cpu")
    def _exec_vector_cpu(self, *args, **kwargs):
        return super()._exec_vector(*args, **kwargs)
    # def _exec_vector_cpu_copy(self, a, bd, mask, block_size):
    #     npt = bd.shape[0]
    #     n = self.X_ADJUSTED.shape[0]
    #     zero_index = None
    #     zero_value = False

    #     # use the desired method to invert the kriging matrix
    #     if self.pseudo_inv:
    #         a_inv = P_INV[self.pseudo_inv_type](a)
    #     else:
    #         a_inv = spl.inv(a)

    #     block_size = block_size  # 分块大小
    #     num_blocks = npt // block_size  # 计算分块数
    #     rem_elements = npt % block_size  # 计算剩余元素个数
        
    #     kvalues = []  # 存储分块结果的列表
    #     sigmasq = []  # 存储分块结果的列表
        
    #     for i in range(num_blocks):
    #         block_start = i * block_size
    #         block_end = (i + 1) * block_size
    #         block_bd = bd[block_start:block_end, :]

    #         zero_value = False  # Reset for each block
    #         # if np.any(np.absolute(block_bd) <= self.eps):
    #         #     zero_value = True
    #         #     zero_index = np.where(np.absolute(block_bd) <= self.eps)

    #         block_b = np.zeros((block_size, n + 1, 1))
    #         block_b[:, :n, 0] = -self.variogram_function(self.variogram_model_parameters, block_bd)
    #         if zero_value and self.exact_values:
    #             block_b[zero_index[0], zero_index[1], 0] = 0.0
    #         block_b[:, n, 0] = 1.0

    #         block_x = np.dot(a_inv, block_b.reshape((block_size, n + 1)).T).reshape(
    #             (1, n + 1, block_size)).T
    #         block_kvalues = np.sum(block_x[:, :n, 0] * self.VALUES, axis=1)
    #         block_sigmasq = np.sum(block_x[:, :, 0] * -block_b[:, :, 0], axis=1)

    #         # 合并块结果
    #         if i == 0:
    #             kvalues = block_kvalues
    #             sigmasq = block_sigmasq
    #         else:
    #             kvalues = np.concatenate((kvalues, block_kvalues), axis=0)
    #             sigmasq = np.concatenate((sigmasq, block_sigmasq), axis=0)
    #         # 释放内存
    #         del block_bd, block_b, block_x, block_kvalues, block_sigmasq
    #         gc.collect()
    #     # 处理剩余元素
    #     if rem_elements > 0:
    #         block_start = num_blocks * block_size
    #         block_end = block_start + rem_elements
    #         block_bd = bd[block_start:block_end, :]

    #         zero_value = False  # Reset for the remaining block
    #         # if np.any(np.absolute(block_bd) <= self.eps):
    #         #     zero_value = True
    #         #     zero_index = np.where(np.absolute(block_bd) <= self.eps)

    #         block_b = np.zeros((rem_elements, n + 1, 1))
    #         block_b[:, :n, 0] = -self.variogram_function(self.variogram_model_parameters, block_bd)
    #         if zero_value and self.exact_values:
    #             block_b[zero_index[0], zero_index[1], 0] = 0.0
    #         block_b[:, n, 0] = 1.0

    #         block_x = np.dot(a_inv, block_b.reshape((rem_elements, n + 1)).T).reshape(
    #             (1, n + 1, rem_elements)).T
    #         block_kvalues = np.sum(block_x[:, :n, 0] * self.VALUES, axis=1)
    #         block_sigmasq = np.sum(block_x[:, :, 0] * -block_b[:, :, 0], axis=1)

    #         kvalues = np.concatenate((kvalues, block_kvalues), axis=0)
    #         sigmasq = np.concatenate((sigmasq, block_sigmasq), axis=0)

    #         del block_bd, block_b, block_x, block_kvalues, block_sigmasq
    #         gc.collect()
    #     return kvalues, sigmasq

    # def _exec_vector_copy(self, a, bd, mask, block_size):
        """Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets."""

    #     npt = bd.shape[0]
    #     n = self.X_ADJUSTED.shape[0]
    #     zero_index = None
    #     zero_value = False

    #     a_cp = cp.asarray(a, dtype=np.float32)
    #     # use the desired method to invert the kriging matrix
    #     if self.pseudo_inv:
    #         a_inv_cp = P_INV[self.pseudo_inv_type](a_cp)
    #     else:
    #         a_inv_cp = cp.linalg.inv(a_cp)

    #     print("a size is" , a.size)
        
    #     print(bd.shape)
        
    #     if self.variogram_model not in variogram_dict_gpu.keys():
    #         raise NotImplementedError(f"variogram_model={self.variogram_model} is not supported in GPU.")
    #     else:
    #         variogram_function_cp = variogram_dict_gpu[self.variogram_model]
        
    #     # 从以下开始分块计算
    #     if npt < 90000: #预测点数小于90000时, 不用分块
    #         print("预测数据量小，不执行分块运算")
    #         # 由于我的bd一般比self.eps大，这一步又很消耗时间，所以默认先注释掉
    #         # if np.any(np.absolute(bd) <= self.eps):
    #         #     zero_value = True
    #         #     zero_index = np.where(np.absolute(bd) <= self.eps)
    #         b = cp.ones((npt, n + 1, 1), dtype=cp.float32)
    #         b[:, :n, 0] = -variogram_function_cp(self.variogram_model_parameters, bd)

    #         if zero_value and self.exact_values:
    #             b[zero_index[0], zero_index[1], 0] = 0.0
            
    #         b[:, n, 0] = 1.0

    #         kvalues, sigmasq = self._exec_vector_unit(bd, b, n, npt, mask, a_inv_cp)


    #     else: #预测点数大于90000时，使用数据分块结合GPU
    #         print("预测数据量大，执行分块运算")
    #         block_size = block_size  # 分块大小

    #         # 按照第二个维度分块
    #         num_blocks = bd.shape[0] // block_size  # 计算分块数
    #         rem_elements = bd.shape[0] % block_size  # 计算剩余元素个数

    #         zero_value = False
    #         zero_row_index = []
    #         zero_col_index = []

    #         kvalues = []  # 存储分块结果的列表
    #         sigmasq = []  # 存储分块结果的列表

    #         for i in range(num_blocks):
                
    #             begin_one = time.time()
    #             block_start = i * block_size
    #             block_end = (i + 1) * block_size
                
    #             block_bd = bd[block_start:block_end, :]
    #             block_b = cp.ones((block_size, n + 1, 1), dtype=cp.float32)
                
    #             # 由于我的bd一般比self.eps大，这一步又很消耗时间，所以默认先注释掉
    #             # if np.any(np.absolute(block_bd) <= self.eps):
    #             #     zero_value = True
    #             #     zero_index = np.where(np.absolute(block_bd) <= self.eps)
    #             begin_three = time.time() #定位计时部分开始
    #             #定位到这一步耗时最多,最后调试发现是结果从cupy转换为numpy耗时最大
    #             block_b[:, :n, 0] = -variogram_function_cp(self.variogram_model_parameters, block_bd)
    #             end_three = time.time() #定位计时部分结束
    #             if zero_value and self.exact_values:
    #                 block_b[zero_index[0], zero_index[1], 0] = 0.0
    #             block_b[:, n, 0] = 1.0
                
    #             begin_two = time.time()
    #             block_kvalues, block_sigmasq = self._exec_vector_unit(block_bd, block_b, n, block_size, mask, a_inv_cp)
    #             end_two = time.time()
                
    #             # 合并块结果
    #             if i == 0:
    #                 kvalues = block_kvalues
    #                 sigmasq = block_sigmasq
    #             else:
    #                 kvalues = np.concatenate((kvalues, block_kvalues), axis=0)
    #                 sigmasq = np.concatenate((sigmasq, block_sigmasq), axis=0)
    #             end_one = time.time()
    #             # print("第",i,"个分块成功",
    #             #       "执行耗时",end_one - begin_one,
    #             #       ",_exec_vector_unit消耗的时间为：",end_two - begin_two,
    #             #       ",定位计时部分消耗的时间为：",end_three - begin_three)
    #             print("第",i,"个分块成功")
    #         # 判断分块后是否有剩余元素        
    #         if rem_elements > 0:
    #             block_start = num_blocks * block_size
    #             block_end = block_start + rem_elements

    #             block_bd = bd[block_start:block_end, :]
    #             block_b = cp.ones((rem_elements, n + 1, 1), dtype=cp.float32)
                
    #             # if np.any(np.absolute(block_bd) <= self.eps):
    #             #     zero_value = True
    #             #     zero_index = np.where(np.absolute(block_bd) <= self.eps)
    #             if zero_value and self.exact_values:
    #                 block_b[zero_index[0], zero_index[1], 0] = 0.0
    #             block_b[:, :n, 0] = -variogram_function_cp(self.variogram_model_parameters, block_bd)
    #             block_b[:, n, 0] = 1.0
                
    #             block_kvalues, block_sigmasq = self._exec_vector_unit(block_bd, block_b, n, rem_elements, mask, a_inv_cp)
    #             kvalues = np.concatenate((kvalues, block_kvalues), axis=0)
    #             sigmasq = np.concatenate((sigmasq, block_sigmasq), axis=0)  
              
    #     cp.get_default_memory_pool().free_all_blocks()
    #     cp.get_default_pinned_memory_pool().free_all_blocks()      
    #     return kvalues, sigmasq
     
    # # def _exec_vector_unit_copy(self, bd, b_cp, n, npt, mask, a_inv_cp):
    #     if (mask).any():
    #             raise NotImplementedError("mask is not supported in GPU.")

        
    #     x_cp = cp.dot(a_inv_cp, b_cp.reshape((npt, n + 1)).T).reshape((1, n + 1, npt)).T
    
    #     values_cp = cp.asarray(self.VALUES, dtype = cp.float32)
    #     # 这一步按值进行了修改（可对照原代码，主要是降低值的影响，结果没用）
    #     # kvalues_cp = cp.sum(x_cp[:, :n, 0] * values_cp * (1-(values_cp/cp.sum(values_cp))), axis=1)
    #     kvalues_cp = cp.sum(x_cp[:, :n, 0] * values_cp, axis=1)
    #     # kvalues_cp = cp.sum(x_cp[:, :n, 0] / values_cp ** 1/2, axis=1) * cp.mean(values_cp) ** 1/2
    #     kvalues = cp.asnumpy(kvalues_cp)
        
    #     sigmasq_cp = cp.sum(x_cp[:, :, 0] * -b_cp[:, :, 0], axis=1)
    #     sigmasq = cp.asnumpy(sigmasq_cp)
        
    #     cp.get_default_memory_pool().free_all_blocks()
    #     cp.get_default_pinned_memory_pool().free_all_blocks()
        
    #     return kvalues, sigmasq
     