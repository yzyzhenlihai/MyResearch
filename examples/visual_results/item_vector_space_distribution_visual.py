import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
import torch


# 如果安装了 umap-learn，建议取消注释下面这行，处理大量数据时速度快几十倍
# import umap 

def extract_embeddings_from_list(data_list):
    """
    从轨迹列表中提取所有时刻的动作 Embedding
    :param data_list: 轨迹列表，格式 [ {'actions': (32, D), ...}, ... ]
    :return: 两个大矩阵 (Total_Steps, Dim) -> (原始, 生成)
    """
    print(f"正在处理 {len(data_list)} 条轨迹数据...")
    
    gen_embs = []
    orig_embs = []
    
    for i, traj in enumerate(data_list):
        # 1. 获取生成动作
        if "actions" in traj:
            act = traj["actions"]
            # 兼容处理：如果是 Tensor 转 numpy，如果是 list 转 numpy
            if isinstance(act, torch.Tensor):
                act = act.cpu().detach().numpy()
            elif isinstance(act, list):
                act = np.array(act)
            gen_embs.append(act)
        
        # 2. 获取原始动作
        if "original_actions" in traj:
            orig = traj["original_actions"]
            if isinstance(orig, torch.Tensor):
                orig = orig.cpu().detach().numpy()
            elif isinstance(orig, list):
                orig = np.array(orig)
            orig_embs.append(orig)

    # 堆叠所有时间步
    # 假设每个 act 是 (32, D)，总共有 N 个轨迹
    # 结果 shape 将是 (N * 32, D)
    if not gen_embs or not orig_embs:
        raise ValueError("提取失败：未在轨迹字典中找到 'actions' 或 'original_actions'")
    
    gen_matrix = np.vstack(gen_embs)
    orig_matrix = np.vstack(orig_embs)
    
    orig_var = np.var(orig_matrix, axis=0).mean()
    gen_var = np.var(gen_matrix, axis=0).mean()

    print(f"原始数据方差: {orig_var:.5f}")
    print(f"生成增强数据方差: {gen_var:.5f}")
    if gen_var > orig_var:
        print("结论：生成数据引入了额外的多样性（好）")
    else:
        print("结论：生成数据比原始数据更收敛（可能发生了模式坍塌或过拟合）")
 

    return orig_matrix, gen_matrix

def visualize_diversity(file_path, sample_n=5000):
    """
    主函数：加载数据 -> 提取 -> 降维 -> 绘图
    """
    # ---------------------------------------------------------
    # 1. 加载数据 (针对 npz 中包含对象的情况)
    # ---------------------------------------------------------
    print(f"正在加载: {file_path}")
    loaded = np.load(file_path, allow_pickle=True)
    
    data_list = loaded['data'].squeeze()
    orig_matrix, gen_matrix = extract_embeddings_from_list(data_list)
    
    print(f"原始动作空间形状: {orig_matrix.shape}") # 预期 (N*32, Dim)
    print(f"生成动作空间形状: {gen_matrix.shape}")

    # ---------------------------------------------------------
    # 3. 随机采样 (避免 t-SNE 跑太慢)
    # ---------------------------------------------------------
    total_points = min(len(orig_matrix), len(gen_matrix))
    n_vis = min(total_points, sample_n)
    
    print(f"随机采样 {n_vis} 个点进行可视化对比...")
    idx_orig = np.random.choice(len(orig_matrix), n_vis, replace=False)
    idx_gen = np.random.choice(len(gen_matrix), n_vis, replace=False)
    
    sample_orig = orig_matrix[idx_orig]
    sample_gen = gen_matrix[idx_gen]

    # ---------------------------------------------------------
    # 4. 联合降维 (Joint t-SNE)
    # ---------------------------------------------------------
    # 必须把两者拼在一起做 t-SNE，才能在同一个坐标系下对比
    combined = np.vstack([sample_orig, sample_gen])
    labels = np.array([0] * n_vis + [1] * n_vis) # 0: Original, 1: Generated

    print("开始运行 t-SNE (这可能需要几分钟)...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
    X_embedded = tsne.fit_transform(combined)

    # ---------------------------------------------------------
    # 5. 绘图分析
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 8), dpi=120)
    
    # 拆分结果
    tsne_orig = X_embedded[labels == 0]
    tsne_gen = X_embedded[labels == 1]

    # 绘制层1：原始分布 (Original) - 灰色背景
    plt.scatter(tsne_orig[:, 0], tsne_orig[:, 1], 
                c='silver', s=20, alpha=0.5, label='Original (Benchmark)', edgecolors='none')
    
    # 绘制层2：生成分布 (Generated) - 红色前景
    # alpha 设置为 0.4 很关键，这样如果生成的点都叠在一起，颜色会变得很深，容易看出 Mode Collapse
    plt.scatter(tsne_gen[:, 0], tsne_gen[:, 1], 
                c='red', s=15, alpha=0.4, label='Generated (GTA)', edgecolors='none')

    plt.title(f'Embedding Diversity Check: Generated vs Original\n(Red points should cover Grey areas, not just cluster in the center)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # 保存与展示
    plt.savefig(f"{os.path.dirname(script_path)}/diversity_analysis_{dataset_name}.png")
    print(f"图表已保存为 {os.path.dirname(script_path)}/diversity_analysis_{dataset_name}.png")
    plt.show()

if __name__ == "__main__":
    dataset_name = 'KuaiRec'  #KuaiRand_Pure
    script_path = os.path.abspath(__file__)
    data_dir = f"{os.path.dirname(os.path.dirname(os.path.dirname(script_path)))}/data/{dataset_name}/data_generated/gta_samples.npz"
    print("当前脚本所在文件夹:", script_path)
    print("数据文件路径:", data_dir)
    try:
        visualize_diversity(data_dir, sample_n=5000)
    except Exception as e:
        import traceback
        traceback.print_exc()