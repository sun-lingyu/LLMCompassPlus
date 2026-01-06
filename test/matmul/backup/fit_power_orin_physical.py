import os
import json
import numpy as np
import pickle
import argparse

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
file_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = f"{file_dir}/power_features_cache.npz"  # 缓存文件路径

PHYSICAL_BOUNDS = {
    # 格式: (Min_Joules, Max_Joules)
    # 1 pJ = 1e-12 J
    
    # FMA: 7nm/8nm MAC 运算极其高效，通常 < 1pJ
    "SysArr FMA":  (1e-13, 5e-12),  # [0.1 pJ,  5.0 pJ] per Op
    
    # Register: 比 SRAM 更省电
    "Reg Access":  (5e-14, 1e-12),  # [0.05 pJ, 1.0 pJ] per Access
    
    # L1 SRAM: 以 Byte 为单位。Cache line 通常 32/64B。
    # 访问一次 L1 (32B) 耗能约 20-50pJ -> 摊薄后 ~1pJ/Byte
    "L1 Access":   (5e-13, 1e-11),  # [0.5 pJ, 10.0 pJ] per Byte
    
    # L2 SRAM: 比 L1 贵 3-5 倍
    "L2 Access":   (2e-12, 5e-11),  # [2.0 pJ, 50.0 pJ] per Byte
    
    # DRAM (LPDDR5): 系统级能耗 (颗粒 + PHY + IO)
    # 业内数据通常在 50-100 pJ/Byte 左右。下限设为 20pJ 比较安全。
    "DRAM Access": (2e-11, 2e-10),  # [20.0 pJ, 200.0 pJ] per Byte
    
    # Static Power: Orin 系列待机功耗
    "Static Power":(1.0,   8.0)    # [1.0 W,  8.0 W]
}

FEATURE_NAMES = ["SysArr FMA", "Reg Access", "L1 Access", "L2 Access", "DRAM Access"]

def physics_guided_fitting(X, y):
    """
    使用 scipy.optimize.minimize 进行物理约束拟合
    """
    print("\n--- Starting Physics-Guided Fitting (L-BFGS-B) ---")
    
    # 1. 自动特征缩放 (Scaling)
    # L-BFGS-B 算法在特征数值都在 1.0 附近时收敛最快
    X_mean = np.mean(X, axis=0)
    X_mean[X_mean == 0] = 1.0
    
    # 我们用 Mean 进行缩放，这样 coefs_scaled 大约在 y_mean 附近
    X_scaled = X / X_mean
    
    # 2. 转换边界 (Bounds) 到缩放后的空间
    # 原始公式: y = X * w + b
    # 缩放公式: y = (X / X_mean) * (w * X_mean) + b
    # 所以: w_scaled = w_real * X_mean
    #      bounds_scaled = bounds_real * X_mean
    
    bounds_list = []
    print(f"{'Component':<15} | {'Physical Bound (Min ~ Max)'}")
    print("-" * 60)
    
    for i, name in enumerate(FEATURE_NAMES):
        low_real, high_real = PHYSICAL_BOUNDS[name]
        
        # 转换到 Scaled 空间
        low_scaled = low_real * X_mean[i]
        high_scaled = high_real * X_mean[i]
        
        bounds_list.append((low_scaled, high_scaled))
        print(f"{name:<15} | [{low_real*1e12:.2f}, {high_real*1e12:.2f}] pJ")

    # Static Power (Intercept) 不需要缩放
    s_low, s_high = PHYSICAL_BOUNDS["Static Power"]
    bounds_list.append((s_low, s_high))
    print(f"{'Static Power':<15} | [{s_low}, {s_high}] W")

    # 3. 损失函数 (MSE)
    def loss_function(params, X_in, y_true):
        w = params[:-1]
        b = params[-1]
        y_pred = X_in @ w + b
        return np.mean((y_true - y_pred) ** 2)

    # 4. 初始猜测 (取边界中点)
    x0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds_list])

    # 5. 求解
    res = minimize(
        loss_function, 
        x0, 
        args=(X_scaled, y), 
        method='L-BFGS-B', 
        bounds=bounds_list,
        options={'ftol': 1e-10, 'maxiter': 5000}
    )

    if not res.success:
        print(f"[Warning] Optimization failed: {res.message}")

    # 6. 还原真实系数
    # w_real = w_scaled / X_mean
    w_scaled = res.x[:-1]
    intercept = res.x[-1]
    
    coefs_real = w_scaled / X_mean
    
    return coefs_real, intercept, res.fun

def plot_fitting_results(y_true, y_pred, feature_names, coefs, intercept, r2, mse):
    # (保持原本的绘图逻辑不变，只修改图表标题以反映新方法)
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('ggplot')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, color='dodgerblue', alpha=0.6, s=50, label='Data')
    min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Ideal')
    ax1.set_title(f'Physics-Guided Model (Samsung 8nm)\n$R^2={r2:.4f}, MSE={mse:.4f}$', fontsize=14)
    ax1.set_xlabel('Measured Power (W)')
    ax1.set_ylabel('Predicted Power (W)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coefficients
    ax2 = axes[1]
    y_pos = np.arange(len(feature_names))
    # 转换为 pJ (皮焦) 展示，更符合直觉
    coefs_pj = coefs * 1e12 
    
    ax2.barh(y_pos, coefs_pj, color='mediumseagreen', edgecolor='k', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names, fontsize=11)
    ax2.set_xlabel('Energy Cost (pJ / Op or Byte)', fontsize=12)
    ax2.set_title('Optimized Energy Cost (Picojoules)', fontsize=14)
    
    for i, v in enumerate(coefs_pj):
        ax2.text(v, i, f' {v:.2f} pJ', va='center', fontweight='bold')

    plt.figtext(0.5, 0.02, f"Static Power = {intercept:.2f} W", ha="center", 
                bbox=dict(facecolor='orange', alpha=0.2))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{file_dir}/power_physics_refined.png", dpi=300)
    print(f"\n[Info] Plot saved to: {file_dir}/power_physics_refined.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("precision", type=str, choices=["fp16", "int8", "int4"])
    args = parser.parse_args()
    CACHE_FILE = CACHE_FILE.replace(".npz", f"_{args.precision}.npz")

    X = None
    y = None
    if os.path.exists(CACHE_FILE):
        print(f"Found cache file: {CACHE_FILE}")
        print("Loading features directly (skipping simulation)...")
        try:
            data = np.load(CACHE_FILE)
            X = data['X']
            y = data['y']
            print(f"Loaded {len(y)} records from cache.")
        except Exception as e:
            print(f"Error loading cache: {e}. Will re-calculate.")
            X = None
    
    if X is None:
        pcb = device_dict["Orin"]
        existing_data = []
        if os.path.exists(f"{file_dir}/cutlass_power_log.json"):
            with open(f"{file_dir}/cutlass_power_log.json", 'r') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    if isinstance(data, list):
                        existing_data = data
                    else:
                        assert False, "cutlass_power_log.json format error"
        else:
            assert False, "cutlass_power_log.json not found"

        X_features = []
        y_power = []

        for record in existing_data:
            M, N, K, precision = record['M'], record['N'], record['K'], record['precision']

            if precision == "fp16":
                activation_data_type=data_type_dict["fp16"]
                weight_data_type=data_type_dict["fp16"]
                intermediate_data_type=data_type_dict["fp32"]
            elif precision == "int8":
                activation_data_type=data_type_dict["int8"]
                weight_data_type=data_type_dict["int8"]
                intermediate_data_type=data_type_dict["int32"]
            elif precision == "int4":
                activation_data_type=data_type_dict["fp16"]
                weight_data_type=data_type_dict["int4"]
                intermediate_data_type=data_type_dict["fp32"]
            
            if precision !=args.precision:
                continue

            model = Matmul(activation_data_type=activation_data_type, weight_data_type=weight_data_type, intermediate_data_type=intermediate_data_type)
            _ = model(
                Tensor([M, K], activation_data_type),
                Tensor([K, N], weight_data_type),
            )
            latency_ms = 1000 * model.compile_and_simulate(pcb, compile_mode="heuristic-GPU")
            runtime_s = latency_ms / 1000.0
            features = [
                model.systolic_array_fma_count / runtime_s,
                model.reg_access_count / runtime_s,
                model.l1_access_size / runtime_s,
                model.l2_access_size / runtime_s,
                model.mem_access_size / runtime_s
            ]
            X_features.append(features)
            y_power.append(record['power'])
        
        if len(X_features) > 0:
            X = np.array(X_features)
            y = np.array(y_power)
            
            print(f"Saving features to cache: {CACHE_FILE}")
            np.savez(CACHE_FILE, X=X, y=y)
        else:
            print("No valid data generated.")

    if X is not None and len(X) > 0:
        # 执行拟合
        real_coefs, static_power, mse = physics_guided_fitting(X, y)
        
        # 预测
        y_pred = X @ real_coefs + static_power
        r2 = r2_score(y, y_pred)
        
        print("-" * 80)
        print(f"Final Results (R^2: {r2:.4f})")
        print(f"{'Component':<15} | {'Optimized (pJ)':<15} | {'Bound Status'}")
        print("-" * 80)
        
        for name, val in zip(FEATURE_NAMES, real_coefs):
            val_pj = val * 1e12
            low, high = PHYSICAL_BOUNDS[name]
            
            status = "OK"
            # 增加一点容差判断是否卡边界
            if abs(val - low) < 1e-15: status = "HIT LOWER BOUND (Ineffective?)"
            if abs(val - high) < 1e-15: status = "HIT UPPER BOUND (Dominant?)"
            
            print(f"{name:<15} | {val_pj:.4f} pJ        | {status}")
            
        print("-" * 80)
        print(f"Static Power    | {static_power:.4f} W")
        
        plot_fitting_results(y, y_pred, FEATURE_NAMES, real_coefs, static_power, r2, mse)