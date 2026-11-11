# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 5:01 下午
# @Author  : Chongming GAO
# @FileName: evaluation.py
import numpy as np
import torch

from src.core.evaluation.metrics import get_diversity, get_novelty
from src.core.evaluation.utils import (
    DEFAULT_MCD_SEGMENT_LENGTH,
    get_feat_dominate_dict,
    get_trajectory_feat_dominate_dict,
)


class Evaluator_Feat:
    """统计旧版 pooled MCD、轨迹等权 MCD 与轨迹内分段 MCD。"""

    def __init__(
        self,
        test_collector_set,
        df_item_val,
        need_transform,
        item_feat_domination,
        lbe_item,
        top_rate,
        draw_bar=False,
        mcd_segment_length=DEFAULT_MCD_SEGMENT_LENGTH,
    ):
        """初始化特征支配度评估器。

        Args:
            test_collector_set (CollectorSet): 包含 FB、NX_0 与 NX_force
                collector 的评估集合。
            df_item_val (pandas.DataFrame): 以原始 item id 为索引的特征表。
            need_transform (bool): 是否需要将内部 item id 反变换为原始 id。
            item_feat_domination (dict | None): 训练数据的类别频次降序表。
            lbe_item (LabelEncoder | None): item id 反变换器。
            top_rate (float): 主导类别累计频率阈值。
            draw_bar (bool): 是否保留旧版类别柱状图数据。
            mcd_segment_length (int): 分段 MCD 的连续窗口长度。

        Raises:
            ValueError: 当分段长度不是正整数时抛出。
        """
        if (
            not isinstance(mcd_segment_length, (int, np.integer))
            or mcd_segment_length <= 0
        ):
            raise ValueError("mcd_segment_length must be a positive integer.")
        self.collector_dict = test_collector_set.collector_dict
        # self.num_items = test_collector_set.env.get_env_attr("mat")[0].shape[1]

        # self.env = env
        self.df_item_val = df_item_val
        self.need_transform = need_transform
        self.item_feat_domination = item_feat_domination
        self.lbe_item = lbe_item
        self.top_rate = top_rate
        self.draw_bar = draw_bar
        self.mcd_segment_length = int(mcd_segment_length)

    def on_epoch_begin(self, epoch):
        """保留 callback 接口；该评估器无需 epoch 初始化。"""
        pass

    def on_train_begin(self):
        """保留 callback 接口；该评估器无需训练初始化。"""
        pass

    def on_train_end(self):
        """保留 callback 接口；该评估器无需训练收尾。"""
        pass

    def on_epoch_end(self, epoch, results=None, **kwargs):
        """从 collector buffer 汇总三个分支的 MCD 指标。

        Args:
            epoch (int): 当前评估 epoch，仅用于兼容 callback 接口。
            results (dict): collector 结果及各分支轨迹起始索引。
            **kwargs: 兼容其他 callback 传入的扩展参数。

        Returns:
            dict: 原始结果加上旧版 pooled、轨迹等权整轨及分段 MCD。
        """

        def get_actions_feat(buffer, indices, draw_bar=False):
            """恢复逐轨迹动作并计算三种 MCD 聚合口径。"""

            num_tests = len(indices)
            live_mat = np.zeros([0, num_tests], dtype=bool)
            act_mat = np.zeros([0, num_tests], dtype=np.int64)

            is_end = np.zeros([num_tests], dtype=bool)

            # indices = results["idxs"]

            while not all(is_end):
                acts = buffer.obs_next[indices][:, 1]  # buffer.act[indices]
                done = buffer.done[indices]

                act_mat = np.vstack([act_mat, acts])
                live_mat = np.vstack([live_mat, ~is_end])

                is_end[done] = True
                indices = buffer.next(indices)

            trajectory_actions = [
                act_mat[:, trajectory_index][live_mat[:, trajectory_index]]
                for trajectory_index in range(num_tests)
            ]
            if self.need_transform:
                trajectory_actions_origin = [
                    self.lbe_item.inverse_transform(actions)
                    for actions in trajectory_actions
                ]
            else:
                trajectory_actions_origin = trajectory_actions
            all_acts_origin = np.concatenate(trajectory_actions_origin)
            feat_dominate_dict = get_feat_dominate_dict(
                self.df_item_val,
                all_acts_origin,
                self.item_feat_domination,
                top_rate=self.top_rate,
                draw_bar=draw_bar,
            )
            feat_dominate_dict.update(
                get_trajectory_feat_dominate_dict(
                    self.df_item_val,
                    trajectory_actions_origin,
                    self.item_feat_domination,
                    top_rate=self.top_rate,
                    segment_length=self.mcd_segment_length,
                )
            )

            return feat_dominate_dict


        results_all = {}
        for name, collector in self.collector_dict.items():
            buffer = collector.buffer
            indices = results[name + "_idxs"] if name != "FB" else results["idxs"]
            feat_dominate_dict = get_actions_feat(
                buffer,
                indices,
                draw_bar=self.draw_bar,
            )
            feat_dominate_dict_k = (
                {
                    name + "_" + metric_name: metric_value
                    for metric_name, metric_value in feat_dominate_dict.items()
                }
                if name != "FB"
                else feat_dominate_dict
            )
            results_all.update(feat_dominate_dict_k)

        results.update(results_all)

        return results
    

class Evaluator_Coverage_Count():
    def __init__(self, test_collector_set, df_item_val, need_transform):
        self.collector_dict = test_collector_set.collector_dict
        self.num_items = test_collector_set.env.get_env_attr("mat")[0].shape[1]

        # self.df_item_val = df_item_val
        self.need_transform = need_transform

    def on_epoch_begin(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, results=None, **kwargs):

        def get_count_results_for_one_collector(buffer):
            live_ind = np.ones([results["n/ep"]], dtype=bool)
            inds = buffer.last_index
            all_acts = []
            res = {}
            while any(live_ind):
                acts = buffer.obs_next[inds][:, 1]  # acts = buffer[inds].act
                # print(acts)
                all_acts.extend(acts)

                live_ind = buffer.prev(inds) != inds
                inds = buffer.prev(inds[live_ind])

            hit_item = len(set(all_acts))
            res["CV"] = hit_item / self.num_items
            res["CV_turn"] = hit_item / len(all_acts)
            return res

        results_all = {}
        for name, collector in self.collector_dict.items():
            buffer = collector.buffer
            res = get_count_results_for_one_collector(buffer)
            res_k = {name + "_" + k: v for k, v in res.items()} if name != "FB" else res
            results_all.update(res_k)

        results.update(results_all)

        return results

class Evaluator_User_Experience():
    def __init__(self, test_collector_set, df_item_val, item_similarity, item_popularity, need_transform, lbe_item):
        self.collector_dict = test_collector_set.collector_dict
        self.num_items = test_collector_set.env.get_env_attr("mat")[0].shape[1]
        self.need_transform = need_transform
        self.lbe_item = lbe_item

        self.df_item_val = df_item_val
        # self.item_similarity = test_collector_set.env.get_env_attr("df_similarity_small")[0]
        # self.item_popularity = test_collector_set.env.get_env_attr("item_popularity")[0]
        self.item_similarity = item_similarity
        self.item_popularity = item_popularity
        
    def on_epoch_begin(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, results=None, **kwargs):

        def get_actions(buffer, indices):

            num_tests = len(indices)
            live_mat = np.zeros([0, num_tests], dtype=bool)
            act_mat = np.zeros([0, num_tests], dtype=bool)
            rew_mat = np.zeros([0, num_tests], dtype=bool)
            is_end = np.zeros([num_tests], dtype=bool)

            while not all(is_end):
                acts = buffer.obs_next[indices][:, 1]  # buffer.act[indices]
                done = buffer.done[indices]
                rews = buffer.rew[indices]

                act_mat = np.vstack([act_mat, acts])
                rew_mat = np.vstack([rew_mat, rews])
                live_mat = np.vstack([live_mat, ~is_end])

                is_end[done] = True
                indices = buffer.next(indices)

            act_mat = act_mat.T
            rew_mat = rew_mat.T
            live_mat = live_mat.T

            all_acts = []
            all_rews = []
            all_acts_origin = []
            if self.need_transform:
                for i, live_arr in enumerate(live_mat):
                    all_acts.append(act_mat[i][live_arr])
                    all_rews.append(rew_mat[i][live_arr])
                    all_acts_origin.append(self.lbe_item.inverse_transform(act_mat[i][live_arr]))
            else:
                for i, live_arr in enumerate(live_mat):
                    all_acts.append(act_mat[i][live_arr])
                    all_rews.append(rew_mat[i][live_arr])
                    all_acts_origin.append(act_mat[i][live_arr])
        
            # if self.need_transform:
            #     all_acts_origin = self.lbe_item.inverse_transform(all_acts)
            # else:
            #     all_acts_origin = all_acts

            return all_acts_origin, all_rews, buffer.obs[indices][:, 0]

        
        def get_results_for_one_collector(actions, rews, users):
            res = {}
            all_div = []
            all_nov = []
            # all_ser = []
            for item_list, rew_list, user in zip(actions, rews, users):
                all_div.append(get_diversity(item_list, self.item_similarity))
                all_nov.append(get_novelty(item_list, self.item_popularity))
                # all_ser.append(get_serendipity(item_list, rew_list, user))
            res["Diversity"] = np.array(all_div).mean()
            res["Novelty"] = np.array(all_nov).mean()
            # res["ser"] = np.array(all_ser).mean()
            return res

        results_all = {}
        for name, collector in self.collector_dict.items():
            buffer = collector.buffer
            indices = results[name + "_idxs"] if name != "FB" else results["idxs"]
            actions, rews, users = get_actions(buffer, indices)
            
            res = get_results_for_one_collector(actions, rews, users)
            res_k = {name + "_" + k: v for k, v in res.items()} if name != "FB" else res
            results_all.update(res_k)

        results.update(results_all)

        return results


def save_model_fn(epoch, policy, model_save_path, optim, state_tracker, is_save=False):
    if not is_save:
        return
    model_save_path = model_save_path[:-3] + "-e{}".format(epoch) + model_save_path[-3:]
    # torch.save(model.state_dict(), model_save_path)
    torch.save({
        'policy': policy.state_dict(),
        'optim_RL': optim[0].state_dict(),
        'optim_state': optim[1].state_dict(),
        'state_tracker': state_tracker.state_dict(),
    }, model_save_path)
