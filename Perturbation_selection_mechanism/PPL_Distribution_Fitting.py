import numpy as np
from scipy import stats
import pandas as pd

# 存放对抗攻击的困惑度值
file_list_gcg = ['put_the_file_path_here',]

# 存放场景攻击的困惑度值
file_list_scene = ['put_the_file_path_here',]

"""
步骤1：数据收集与预处理
"""
scene_samples = np.concatenate([
    pd.read_csv(f).iloc[:, 1].values for f in file_list_scene
])

gcg_samples = np.concatenate([
    pd.read_csv(f).iloc[:, 1].values for f in file_list_gcg
])

def remove_outliers(data, factor=1.5):
    """
    使用IQR方法去除数据中的异常值

    参数:
        data: numpy数组或类似数组结构，待处理的数据
        factor: float, 控制异常值检测范围的乘数因子，默认为1.5

    返回:
        过滤后的数据，排除了低于Q1-factor*IQR或高于Q3+factor*IQR的值
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

# 对gcg_samples数据进行异常值清洗
gcg_clean = remove_outliers(gcg_samples)

# 对scene_samples数据进行异常值清洗
scene_clean = remove_outliers(scene_samples)



"""
步骤2：分布拟合与验证
"""
def process_method_data(clean_data):
    """
    处理PPL数据文件，进行异常值去除、对数转换和分布拟合

    参数:
        file_path: str, 包含PPL数据的CSV文件路径
        method_name: str, 方法名称，仅用于标识

    返回:
        包含以下键的字典:
        - params: 原始数据的(均值, 标准差)元组
        - dist: 拟合的对数正态分布对象
        - ci: 原始空间的95%置信区间元组
    """
    # 关键步骤：对数转换处理PPL数据
    log_data = np.log(clean_data)  # PPL≥1，取对数安全

    # 在对数空间拟合正态分布
    log_params = stats.norm.fit(log_data)
    log_dist = stats.norm(*log_params)
    log_ci = log_dist.interval(0.95)

    # 将对数空间的置信区间转换回原始空间
    ci_original = (np.exp(log_ci[0]), np.exp(log_ci[1]))

    # 创建原始空间的对数正态分布对象
    dist_original = stats.lognorm(s=log_params[1], scale=np.exp(log_params[0]))

    return {
        'params': (np.mean(clean_data), np.std(clean_data)),
        'dist': dist_original,
        'ci': ci_original
    }

# 使用process_method_data处理GCG数据
gcg_result = process_method_data(gcg_clean)
dist_gcg = gcg_result['dist']
ci_gcg = gcg_result['ci']

# 打印基本信息
print(f"GCG方法:")
print(f"  均值: {gcg_result['params'][0]:.2f}, 标准差: {gcg_result['params'][1]:.2f}")
print(f"  95%置信区间: [{ci_gcg[0]:.2f}, {ci_gcg[1]:.2f}]\n")

# 使用process_method_data处理场景数据（虽然场景数据更适合正态分布，但保持一致性）
scene_result = process_method_data(scene_clean)
dist_scene = scene_result['dist']
ci_scene = scene_result['ci']

# 打印基本信息
print(f"场景方法:")
print(f"  均值: {scene_result['params'][0]:.2f}, 标准差: {scene_result['params'][1]:.2f}")
print(f"  95%置信区间: [{ci_scene[0]:.2f}, {ci_scene[1]:.2f}]\n")

"""
步骤3：阈值优化计算 
"""
# 优化目标函数 - 添加权重系数
def objective(theta, alpha=0.5):
    """
    优化目标函数，用于平衡两类错误（GCG误判和场景误判）的加权组合

    参数:
        theta: float, 待评估的阈值参数
        alpha: float, 权衡系数(0-1)，默认0.5表示平等对待两类错误
              - 当alpha接近0时，优化目标更关注减少GCG误判（即dist_gcg的累积概率）
              - 当alpha接近1时，优化目标更关注减少场景误判（即1-dist_scene的累积概率）

    返回值:
        float: 加权后的目标函数值
    """
    return alpha * dist_gcg.cdf(theta) + (1-alpha) * (1 - dist_scene.cdf(theta))

# 设置Bootstrap抽样次数
n_bootstraps = 1000

# 使用scipy优化方法寻找最优阈值
from scipy.optimize import brentq, minimize_scalar

"""
主优化流程：
1. 首选brentq方法寻找两类错误概率相等的交叉点
2. 若失败则改用最小化目标函数的方法
"""
try:
    # 使用Brentq方法寻找两类分布的交点
    optimal_theta = brentq(
        lambda t: dist_gcg.cdf(t) - (1 - dist_scene.cdf(t)),  # 寻找两类错误相等的点
        a=np.percentile(scene_clean, 95),  # 搜索区间下限：场景数据的95百分位
        b=np.percentile(gcg_clean, 5),    # 搜索区间上限：GCG数据的5百分位
        xtol=1e-4                         # 容差精度
    )
    print(f"优化后的最优阈值: {optimal_theta:.2f}")
except ValueError:
    # Brentq失败时的备选方案：在限定范围内最小化目标函数
    optimal_theta = minimize_scalar(
        objective,
        bounds=(np.percentile(scene_clean, 90), np.percentile(gcg_clean, 10)),  # 更保守的边界
        method='bounded'
    ).x
    print(f"次优阈值: {optimal_theta:.2f}")



"""
步骤4：置信区间计算与验证 
"""
# 使用更高效的并行Bootstrap
from joblib import Parallel, delayed

def bootstrap_iteration():
    gcg_resample = np.random.choice(gcg_clean, size=len(gcg_clean), replace=True)
    scene_resample = np.random.choice(scene_clean, size=len(scene_clean), replace=True)

    gcg_res = process_method_data(gcg_resample)
    scene_res = process_method_data(scene_resample)

    try:
        return brentq(
            lambda t: gcg_res['dist'].cdf(t) - (1 - scene_res['dist'].cdf(t)),
            a=np.percentile(scene_resample, 95),
            b=np.percentile(gcg_resample, 5),
            xtol=1e-3
        )
    except:
        return None

theta_samples = Parallel(n_jobs=-1)(
    delayed(bootstrap_iteration)() for _ in range(n_bootstraps)
)
theta_samples = [x for x in theta_samples if x is not None]  # 过滤失败案例

theta_ci = np.percentile(theta_samples, [2.5, 97.5])
print(f"优化后的阈值置信区间: [{theta_ci[0]:.2f}, {theta_ci[1]:.2f}]")


"""
步骤5：可视化验证
"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))

# 动态调整坐标范围
x_min = max(min(scene_clean)*0.8, 1)  # 确保最小>0
x_max = min(max(gcg_clean)*1.2, 1e6)  # 设置上限
x = np.logspace(np.log10(x_min), np.log10(x_max), 1000)

# 绘制分布曲线（加粗线条）
plt.plot(x, dist_gcg.pdf(x), 'r-', lw=3, label='GCG (Lognormal)')
plt.plot(x, dist_scene.pdf(x), 'b-', lw=3, label='Scene (Lognormal)')

# 增强可视化元素
plt.axvspan(ci_scene[0], ci_scene[1], color='blue', alpha=0.1, label='Scene 95% CI')
plt.axvspan(ci_gcg[0], ci_gcg[1], color='red', alpha=0.1, label='GCG 95% CI')

# 优化阈值标记（加粗虚线）
plt.axvline(optimal_theta, color='k', ls='--', lw=3, label=f'Threshold (θ={optimal_theta:.2f})')
plt.fill_betweenx([0, max(dist_gcg.pdf(x))], theta_ci[0], theta_ci[1], color='gray', alpha=0.2)

# 添加统计信息（增大字号）
plt.text(optimal_theta*1.1, max(dist_gcg.pdf(x))*0.8,
         f'False Positive: {dist_scene.cdf(optimal_theta)*100:.1f}%\n'
         f'False Negative: {(1-dist_gcg.cdf(optimal_theta))*100:.1f}%',
         bbox=dict(facecolor='white', alpha=0.8),
         fontsize=24)  # 从默认10增大到14

# 设置全局字体大小
plt.rcParams.update({'font.size': 24})

# 坐标轴标签和标题（增大字号）
plt.xlabel('Perplexity (PPL)', fontsize=24)
plt.ylabel('Probability Density', fontsize=24)
#刻度值字体大小设置（x轴和y轴同时设置）
plt.tick_params(labelsize=24)
# plt.title('Optimized Threshold Detection between GCG and Scene Attacks',
#           fontsize=18, pad=20)

# 图例设置（增大字号和边框）
plt.legend(loc='upper right', fontsize=24, framealpha=1)
plt.grid(True, which="both", ls="--", alpha=0.3)

plt.xscale('log')
plt.tight_layout()
plt.savefig('optimized_threshold.PDF', dpi=600, bbox_inches='tight')
