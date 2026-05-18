import numpy as np
import os
import sys
from hydra.utils import instantiate
from hydra import initialize, compose
import argparse
from omegaconf import OmegaConf
import torch
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if root_path not in sys.path:
    sys.path.append(root_path)
from examples.policy.policy_utils import prepare_user_model


def extract_embeddings_from_list(data_list):
    """
    从轨迹列表中提取所有时刻的动作 Embedding
    :param data_list: 轨迹列表，格式 [ {'actions': (32, D), ...}, ... ]
    :return: 两个大矩阵 (Total_Steps, Dim) -> (原始, 生成)
    """
    print(f"正在处理 {len(data_list)} 条轨迹数据...")
    
    gen_embs = []
    orig_embs = []
    user_ids = []
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
        if "user_id" in traj:
            uid = traj["user_id"]
            if isinstance(uid, torch.Tensor):
                uid = uid.cpu().detach().numpy()
            elif isinstance(uid, list):
                uid = np.array(uid)
            user_ids.append(uid)
    # 堆叠所有时间步
    # 假设每个 act 是 (32, D)，总共有 N 个轨迹
    # 结果 shape 将是 (N * 32, D)
    if not gen_embs or not orig_embs:
        raise ValueError("提取失败：未在轨迹字典中找到 'actions' 或 'original_actions'")
    
    gen_matrix = np.vstack(gen_embs)
    orig_matrix = np.vstack(orig_embs)
    user_ids = np.vstack(user_ids).squeeze() # 去掉维度为1的维度

    return orig_matrix, gen_matrix, user_ids

def map_vector_to_discret(generated_actions, all_item_embeddings, device="cpu", batch_size=1024):
    """
    generated_actions: (N, D)  生成的动作向量
    all_item_embeddings: (M, D)  所有物品的 Embedding 矩阵 (索引即 ItemID)
    """
    
    num_samples = generated_actions.shape[0]
    print(f"开始映射 {num_samples} 个向量到 {all_item_embeddings.shape[0]} 个物品...")

    # 将输入数据移动到指定设备
    generated_actions = torch.tensor(generated_actions, dtype=torch.float32, device=device)
    all_item_embeddings = torch.tensor(all_item_embeddings, dtype=torch.float32, device=device)
    mapped_ids = [] # 存每个向量对应的itemID
    with torch.no_grad():
        for i in range(0, num_samples, batch_size): # 分批处理，防止显存爆炸
            batch_gen = generated_actions[i : i + batch_size] # (Batch, D)
            
            # (Batch, D) @ (D, M) -> (Batch, M)
            # 这一步一次性算出了 Batch 里所有向量和所有物品的相似度
            similarity = torch.matmul(batch_gen, all_item_embeddings.T)

            # values是相似度，indices就是对应的 ItemID
            values, indices = torch.topk(similarity, k=1, dim=1)
            
            mapped_ids.append(indices.cpu())

    all_mapped_ids = torch.cat(mapped_ids, dim=0).squeeze()
    return all_mapped_ids

def plot_popularity_distribution(origin_ids, gen_ids, title="Popularity Distribution"):
    """
    绘制流行度分布对比图 (Rank-Frequency Plot)
    :param origin_ids: 原始数据的 Item ID 列表 (1D array/list/tensor)
    :param gen_ids: 生成数据的 Item ID 列表 (1D array/list/tensor)
    """
    # --- 1. 数据预处理 ---
    # 转为 list
    if hasattr(origin_ids, 'tolist'): origin_ids = origin_ids.tolist()
    if hasattr(gen_ids, 'tolist'): gen_ids = gen_ids.tolist()
    
    # 统计频次 (Popularity Calculation)
    origin_counts = Counter(origin_ids)
    gen_counts = Counter(gen_ids)
    
    # --- 2. 确定 X 轴顺序 (关键步骤) ---
    # 获取所有出现过的物品集合
    all_items = list(set(origin_counts.keys()) | set(gen_counts.keys()))
    
    # 【核心】：按照“原始数据”的流行度降序排列
    # 如果原始数据没出现过(generated novel items)，默认频率为0，排在最后
    sorted_items = sorted(all_items, key=lambda x: origin_counts.get(x, 0), reverse=True)
    
    # --- 3. 准备 Y 轴数据 ---
    # 提取对应的频次
    y_origin = np.array([origin_counts.get(i, 0) for i in sorted_items])
    y_gen = np.array([gen_counts.get(i, 0) for i in sorted_items])
    
    # 归一化 (Normalization)
    # 因为生成数据的总量可能和原始数据不一样，比较“概率密度”比比较“绝对次数”更公平
    y_origin_prob = y_origin / y_origin.sum()
    y_gen_prob = y_gen / y_gen.sum()
    
    # 生成 X 轴排名 (Rank)
    x_rank = np.arange(1, len(sorted_items) + 1)
    
    # --- 4. 绘图 (推荐双对数坐标) ---
    plt.figure(figsize=(10, 6), dpi=120)
    
    # 画线
    # 原始分布通常符合幂律分布 (一条直线下降)
    plt.loglog(x_rank, y_origin_prob, label='Original Data', 
               color='black', linestyle='--', linewidth=2, alpha=0.6)
    
    # 生成分布
    plt.loglog(x_rank, y_gen_prob, label='Generated Data (GTA)', 
               color='red', linewidth=1.5, alpha=0.8)
    
    plt.title(f"{title} (Log-Log Scale)")
    plt.xlabel("Item Rank (Sorted by Original Popularity)")
    plt.ylabel("Normalized Frequency (Probability)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 标注区域
    plt.axvline(x=len(x_rank)*0.2, color='green', linestyle=':', alpha=0.5)
    plt.text(len(x_rank)*0.01, min(y_origin_prob)*10, "Head (Hot)", color='green')
    plt.text(len(x_rank)*0.5, min(y_origin_prob)*10, "Tail (Cold)", color='green')
    plt.savefig(f"{os.path.dirname(os.path.dirname(script_path))}/visual_results/{title}.png")
    plt.show()


def plot_sparsity_filling(origin_users, origin_items, gen_users, gen_items):
    # 1. 构造交互集合 (User, Item)
    # 假设输入都是列表或数组
    orig_pairs = set(zip(origin_users, origin_items))
    gen_pairs = set(zip(gen_users, gen_items))
    
    # 2. 找出“纯新增”的交互 (Generated - Original)
    new_pairs = gen_pairs - orig_pairs
    print(f"原始唯一交互数: {len(orig_pairs)}")
    print(f"生成唯一交互数: {len(gen_pairs)}")
    print(f"成功挖掘的新交互数 (Filling Sparsity): {len(new_pairs)}")
    
    # 3. 统计每个用户的表现
    # df_orig 统计原始交互数
    df_orig = pd.DataFrame(list(orig_pairs), columns=['u', 'i'])
    user_counts_orig = df_orig['u'].value_counts()
    
    # df_new 统计新增交互数
    df_new = pd.DataFrame(list(new_pairs), columns=['u', 'i'])
    user_counts_new = df_new['u'].value_counts()
    
    # 4. 对齐用户并排序 (关键：按原始稀疏度排序)
    all_users = list(set(user_counts_orig.index) | set(user_counts_new.index))
    # 按原始交互次数从小到大排序 (Sparse Users First)
    all_users.sort(key=lambda u: user_counts_orig.get(u, 0))
    
    y_orig = [user_counts_orig.get(u, 0) for u in all_users]
    y_new = [user_counts_new.get(u, 0) for u in all_users]
    
    # 5. 绘图 (Stacked Area 或 Bar)
    x = range(len(all_users))
    
    plt.figure(figsize=(12, 6))
    
    # 绘制原始交互 (底部)
    plt.bar(x, y_orig, color='gray', label='Existing Interactions', width=1.0, alpha=0.6)
    
    # 绘制新增交互 (叠加在上面)
    plt.bar(x, y_new, bottom=y_orig, color='red', label='New Interactions (Filled by GTA)', width=1.0, alpha=0.8)
    
    plt.xlabel("Users (Sorted by Original Sparsity: Sparse -> Dense)")
    plt.ylabel("Number of Unique Interactions")
    plt.title("Sparsity Filling Effect: New Interactions per User")
    plt.legend()
    
    # 只显示部分 X 轴标签防止拥挤
    plt.xticks([]) 
    
    # 添加注释
    plt.text(len(x)*0.1, max(y_orig)*0.8, "Sparse Users Region", fontsize=12, color='blue', ha='center')
    plt.arrow(len(x)*0.1, max(y_orig)*0.7, 0, -max(y_orig)*0.2, head_width=len(x)*0.02, color='blue')
    plt.show()


if __name__ == "__main__":

    dataset_name = 'KuaiRec'  #KuaiRand_Pure

    print("当前工作目录：", os.getcwd())
    with initialize(config_path="../../configs"):
        cfg = compose(config_name="gta.yaml")
    
    cfg_dict = OmegaConf.to_container(cfg.RSEnvConfig, resolve=True)
    rs_env_args = argparse.Namespace(**cfg_dict)
    ensemble_models = prepare_user_model(rs_env_args)
    """获得真实物品嵌入"""
    saved_emb = ensemble_models.load_user_item_embedding(freeze_emb=rs_env_args.freeze_emb) # KuaiEnv item(10728,41) user(7176, 8)
    user_emb = saved_emb['feat_user'].weight.cpu().numpy()
    item_emb = saved_emb['feat_item'].weight.cpu().numpy()
    print("item_emb shape:", item_emb.shape) # item_emb shape: (10728, 41)
    """获得数据增强item的向量"""
    
    script_path = os.path.abspath(__file__)
    data_dir = f"{os.path.dirname(os.path.dirname(os.path.dirname(script_path)))}/data/{dataset_name}/data_generated/gta_samples.npz"
    generated_data = np.load(data_dir, allow_pickle=True)
    generated_data = generated_data['data'].squeeze()

    orig_matrix, gen_matrix, user_ids = extract_embeddings_from_list(generated_data)
    
    print("原始数据动作矩阵 shape:", orig_matrix.shape)
    print("生成数据动作矩阵 shape:", gen_matrix.shape)

    """对生成的动作向量进行离散化映射"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    origin_item_mapped_ids = map_vector_to_discret(orig_matrix, item_emb, device=device)
    generated_item_mapped_ids = map_vector_to_discret(gen_matrix, item_emb, device=device)

    print("原始数据映射后 item ID 示例:", origin_item_mapped_ids[:10])
    print("生成数据映射后 item ID 示例:", generated_item_mapped_ids[:10])

    """绘制流行度分布对比图"""
    plot_popularity_distribution(origin_item_mapped_ids, generated_item_mapped_ids,
                                 title=f"Item Popularity Distribution Comparison on {dataset_name}")
    
    """绘制稀疏性填充效果图"""
    # 获得原始数据的用户-物品对

    # 获得生成数据的用户-物品对
    gen_users = user_ids
    gen_items = generated_item_mapped_ids
    plot_sparsity_filling(origin_users, origin_items, gen_users, gen_items)