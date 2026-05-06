#!/usr/bin/env python3
"""结果可视化"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_comparison(results_file="results_accuracy.json"):
    """绘制准确率对比图"""
    with open(results_file, "r") as f:
        results = json.load(f)
    
    methods = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in methods]
    top3_accuracies = [results[m]["top3_accuracy"] * 100 for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Top-1 Accuracy')
    bars2 = ax.bar(x + width/2, top3_accuracies, width, label='Top-3 Accuracy')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fault Localization Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=150)
    plt.show()
    print("Saved to accuracy_comparison.png")


def plot_efficiency_curve(results_file="results_efficiency.json"):
    """绘制效率曲线"""
    with open(results_file, "r") as f:
        results = json.load(f)
    
    max_steps = sorted([int(k) for k in results.keys()])
    accuracies = [results[str(s)]["accuracy"] * 100 for s in max_steps]
    times = [results[str(s)]["avg_time"] for s in max_steps]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 准确率 vs 步数上限
    ax1.plot(max_steps, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Max Steps')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs Max Steps')
    ax1.grid(True)
    
    # 时间 vs 步数上限
    ax2.plot(max_steps, times, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Max Steps')
    ax2.set_ylabel('Average Time (s)')
    ax2.set_title('Time vs Max Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("efficiency_curve.png", dpi=150)
    plt.show()
    print("Saved to efficiency_curve.png")


def plot_ablation_results(results_file="results_ablation.json"):
    """绘制消融实验结果"""
    with open(results_file, "r") as f:
        results = json.load(f)
    
    methods = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in methods]
    top3_accuracies = [results[m]["top3_accuracy"] * 100 for m in methods]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Top-1 Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, top3_accuracies, width, label='Top-3 Accuracy', color='coral')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Ablation Study: Impact of Different Components')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=results["Full Model"]["accuracy"] * 100, color='green', linestyle='--', 
               label=f'Full Model Baseline: {results["Full Model"]["accuracy"]*100:.1f}%')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("ablation_results.png", dpi=150)
    plt.show()
    print("Saved to ablation_results.png")


if __name__ == "__main__":
    print("Generating visualizations...")
    plot_accuracy_comparison()
    plot_efficiency_curve()
    plot_ablation_results()
    print("Done!")