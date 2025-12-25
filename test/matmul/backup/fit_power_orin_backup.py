import os
import json
import numpy as np
import pickle
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from software_model.matmul import Matmul
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import device_dict
file_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = f"{file_dir}/xxx.npz"  # 缓存文件路径

def plot_fitting_results(y_true, y_pred, feature_names, coefs, intercept, r2, mse):
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('ggplot')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 图1: 预测对比 ---
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, color='navy', alpha=0.6, s=60, label='Records')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
    ax1.set_title(f'Physical Power Model (NNLS)\n$R^2={r2:.4f}, MSE={mse:.4f}$', fontsize=14)
    ax1.set_xlabel('Measured Power (W)', fontsize=12)
    ax1.set_ylabel('Predicted Power (W)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 图2: 物理能耗系数 (J/op) ---
    ax2 = axes[1]
    y_pos = np.arange(len(feature_names))
    
    # 颜色：全是绿色，因为不仅positive=True保证了非负
    bars = ax2.barh(y_pos, coefs, color='forestgreen', alpha=0.8, edgecolor='k')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names, fontsize=12)
    ax2.set_xlabel('Energy Cost (Joules / Op or Byte)', fontsize=12)
    ax2.set_title('Estimated Energy Per Operation (Must be >= 0)', fontsize=14)
    
    # 在柱状图旁标注科学计数法数值
    for i, v in enumerate(coefs):
        ax2.text(v, i, f' {v:.2e} J', va='center', fontsize=10, fontweight='bold')

    # 添加静态功耗说明
    plt.figtext(0.5, 0.02, f"Static Power (Intercept) = {intercept:.4f} W", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) # 留出底部空间给文字
    save_path = f"{file_dir}/power_nnls_fitting.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Info] Plot saved to: {save_path}")

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
            latency_ms = 1000 * (model.compile_and_simulate(pcb, compile_mode="heuristic-GPU") + 2773 / pcb.compute_module.clock_freq)
            runtime_s = latency_ms / 1000.0
            print("-" * 10)
            print("M N K:", M, N, K)
            print(f"latency_ms: {latency_ms}\nmodel.systolic_array_fma_count: {model.systolic_array_fma_count}\nmodel.reg_access_count: {model.reg_access_count}\nmodel.l1_access_size: {model.l1_access_size}\nmodel.l2_access_size: {model.l2_access_size}\nmodel.mem_access_size: {model.mem_access_size}")
            features = [
                model.systolic_array_fma_count / runtime_s,
                model.reg_access_count / runtime_s,
                model.l1_access_size / runtime_s,
                model.l2_access_size / runtime_s,
                model.mem_access_size / runtime_s
            ]
            X_features.append(features)
        
        if len(X_features) > 0:
            X = np.array(X_features)
            y = np.array(y_power)
            
            # print(f"Saving features to cache: {CACHE_FILE}")
            # np.savez(CACHE_FILE, X=X, y=y)
        else:
            print("No valid data generated.")

    if X is not None and len(X) > 0:
        feature_names = ["SysArr FMA", "Reg Access", "L1 Access", "L2 Access", "DRAM Access"]

        corr = pd.DataFrame(X, columns=feature_names).corr()
        print(corr)
        sns.heatmap(corr, annot=True)
        plt.savefig("corr.png", dpi=300)

        print("\n--- Fitting with Linear Regression (Positive=True) ---")
        
        # 标准化
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X)

        # 拟合 (NNLS)
        nnls_model = LinearRegression(positive=True, fit_intercept=True)
        nnls_model.fit(X_scaled, y)
        y_pred = nnls_model.predict(X_scaled)
        
        # 评估
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        print(nnls_model.coef_)
        print(nnls_model.intercept_)

        # 还原系数
        real_coefs = nnls_model.coef_ / scaler.scale_
        static_power = nnls_model.intercept_

        print(f"R^2 Score: {r2:.4f}")
        print(f"MSE:       {mse:.4f}")
        print("-" * 65)
        print(f"{'Component':<15} | {'Energy Cost (J/op or J/byte)':<30} | {'Status'}")
        print("-" * 65)
        
        for name, val in zip(feature_names, real_coefs):
            status = "OK" if val > 1e-15 else "ZERO"
            print(f"{name:<15} | {val:.6e}                    | {status}")
            
        print("-" * 65)
        print(f"Static Power:    {static_power:.4f} W")

        # 绘图
        plot_fitting_results(y, y_pred, feature_names, real_coefs, static_power, r2, mse) # 确保你有这个函数
        
    else:
        print("No data available to fit.")