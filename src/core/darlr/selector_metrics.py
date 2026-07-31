"""DARLR selector 训练指标的按 epoch 聚合与回调接口。"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Mapping


METRIC_PREFIX = "selector/train"
"""SwanLab 中 selector 训练指标使用的统一前缀。"""


class WeightedScalarAccumulator:
    """按样本数聚合 selector 的标量训练指标。

    普通指标按权重计算均值；配对的 ``*_mean`` 与 ``*_std`` 使用一、
    二阶矩合并为整个 epoch 的总体标准差；名称以 ``_min`` 或 ``_max``
    结尾的指标分别保留整个 epoch 的最小值或最大值。非有限值不会污染
    聚合结果，但会累计到 ``nonfinite_value_count``，用于定位数值异常。
    """

    def __init__(self) -> None:
        """初始化空的指标聚合器。

        Returns:
            None: 聚合状态保存在实例内部。
        """

        self.reset()

    def reset(self) -> None:
        """清空当前 epoch 的全部统计状态。

        Returns:
            None: 调用后所有计数与聚合值归零。
        """

        self._weighted_sums: DefaultDict[str, float] = defaultdict(float)
        self._weights: DefaultDict[str, float] = defaultdict(float)
        self._second_moment_sums: DefaultDict[str, float] = defaultdict(float)
        self._moment_weights: DefaultDict[str, float] = defaultdict(float)
        self._minimums: Dict[str, float] = {}
        self._maximums: Dict[str, float] = {}
        self._minibatch_count = 0
        self._sample_count = 0
        self._nonfinite_value_count = 0

    def update(self, metrics: Mapping[str, float], weight: int) -> None:
        """加入一个 minibatch 的 selector 指标。

        Args:
            metrics (Mapping[str, float]): 指标名到标量值的映射。
            weight (int): 当前 minibatch 的样本数，必须大于 0。

        Returns:
            None: 指标被原地累积。

        Raises:
            ValueError: 当 ``weight`` 不为正数时抛出。
            TypeError: 当指标值不能转换为浮点数时抛出。
        """

        if weight <= 0:
            raise ValueError("weight must be a positive integer.")

        self._minibatch_count += 1
        self._sample_count += int(weight)
        for key, raw_value in metrics.items():
            value = float(raw_value)
            if not math.isfinite(value):
                self._nonfinite_value_count += 1
                continue
            if key.endswith("_std"):
                metric_prefix = key.removesuffix("_std")
                mean_key = f"{metric_prefix}_mean"
                if mean_key in metrics:
                    mean_value = float(metrics[mean_key])
                    if not math.isfinite(mean_value):
                        continue
                    self._second_moment_sums[metric_prefix] += (
                        value * value + mean_value * mean_value
                    ) * weight
                    self._moment_weights[metric_prefix] += weight
                    continue
            if key.endswith("_min"):
                self._minimums[key] = min(self._minimums.get(key, value), value)
            elif key.endswith("_max"):
                self._maximums[key] = max(self._maximums.get(key, value), value)
            else:
                self._weighted_sums[key] += value * weight
                self._weights[key] += weight

    def summary(self, prefix: str = METRIC_PREFIX) -> Dict[str, float]:
        """生成可直接上传到 SwanLab 的 epoch 指标。

        Args:
            prefix (str): 指标名前缀；默认使用 ``selector/train``。

        Returns:
            Dict[str, float]: 已聚合的标量指标。即使当前 epoch 没有
            selector 更新，也会返回计数类指标。
        """

        normalized_prefix = prefix.rstrip("/")
        result = {
            f"{normalized_prefix}/{key}": (
                self._weighted_sums[key] / self._weights[key]
            )
            for key in self._weighted_sums
            if self._weights[key] > 0
        }
        result.update(
            {
                f"{normalized_prefix}/{key}": value
                for key, value in self._minimums.items()
            }
        )
        result.update(
            {
                f"{normalized_prefix}/{key}": value
                for key, value in self._maximums.items()
            }
        )
        for metric_prefix, second_moment_sum in self._second_moment_sums.items():
            moment_weight = self._moment_weights[metric_prefix]
            mean_key = f"{metric_prefix}_mean"
            mean_weight = self._weights.get(mean_key, 0.0)
            if moment_weight <= 0 or mean_weight <= 0:
                continue
            global_mean = self._weighted_sums[mean_key] / mean_weight
            variance = max(
                second_moment_sum / moment_weight - global_mean * global_mean,
                0.0,
            )
            result[f"{normalized_prefix}/{metric_prefix}_std"] = math.sqrt(variance)
        result[f"{normalized_prefix}/minibatch_count"] = float(self._minibatch_count)
        result[f"{normalized_prefix}/sample_count"] = float(self._sample_count)
        result[f"{normalized_prefix}/nonfinite_value_count"] = float(
            self._nonfinite_value_count
        )
        return result


class SelectorTrainingMetricsCallback:
    """把 DARLR selector 的 epoch 训练统计交给现有 SwanLab 回调链路。

    该回调只负责 epoch 边界管理。具体指标由 ``DARLRPolicy.learn()``
    在每个 minibatch 更新，并由策略的公开诊断接口导出。
    """

    def __init__(self, selector_policy: Any) -> None:
        """绑定包含 selector 指标接口的底层 DARLR 策略。

        Args:
            selector_policy (Any): 必须提供
                ``reset_selector_training_metrics()`` 与
                ``get_selector_training_metrics()`` 方法的策略对象。

        Raises:
            TypeError: 当传入对象缺少必要接口时抛出。
        """

        required_methods = (
            "reset_selector_training_metrics",
            "get_selector_training_metrics",
        )
        missing_methods = [
            method_name
            for method_name in required_methods
            if not callable(getattr(selector_policy, method_name, None))
        ]
        if missing_methods:
            raise TypeError(
                "selector_policy misses metric methods: "
                + ", ".join(sorted(missing_methods))
            )
        self.selector_policy = selector_policy

    def on_train_begin(self, **kwargs: Any) -> None:
        """在训练开始时清空潜在的历史统计。

        Args:
            **kwargs (Any): 与通用回调接口兼容的扩展参数。

        Returns:
            None: 只重置策略内部聚合器。
        """

        self.selector_policy.reset_selector_training_metrics()

    def on_train_end(self, **kwargs: Any) -> None:
        """兼容通用训练回调接口。

        Args:
            **kwargs (Any): 与通用回调接口兼容的扩展参数。

        Returns:
            None: 训练结束时不执行额外操作。
        """

        return None

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        """在每个 epoch 开始前重置 selector 聚合器。

        Args:
            epoch (int): 当前 epoch 编号，必须由 trainer 提供。
            **kwargs (Any): 与通用回调接口兼容的扩展参数。

        Returns:
            None: 只重置当前 epoch 的统计。
        """

        del epoch, kwargs
        self.selector_policy.reset_selector_training_metrics()

    def on_epoch_end(
        self,
        epoch: int,
        results: Any = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """导出当前 epoch 的 selector 标量指标。

        Args:
            epoch (int): 当前 epoch 编号。
            results (Any): 测试结果；selector 训练统计不依赖该参数。
            **kwargs (Any): 与通用回调接口兼容的扩展参数。

        Returns:
            Dict[str, float]: 可由 trainer 直接上传 SwanLab 的指标字典。
        """

        del epoch, results, kwargs
        return self.selector_policy.get_selector_training_metrics()
