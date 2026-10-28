"""仅依赖预计算矩阵与 embedding 的轻量用户模型适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class PrecomputedModelArtifacts:
    """向模拟环境暴露预计算用户模型产物而不实例化 DeepFM。

    ROLeR 的策略训练只会读取预测奖励矩阵和验证集用户/物品 embedding，
    不会调用 DeepFM 前向计算。使用该适配器可避免加载无关模型，并使仅保存了
    ``matsPre`` 与验证 embedding 的 KuaiRand 也能进入训练流程。

    Args:
        model_root: 用户模型产物目录，例如
            ``saved_models/KuaiRand-v0/DeepFM``。
        message: 产物文件名中的实验消息，通常为 ``pointneg``。
        scratch_action_embedding: 是否用确定性随机物品 embedding 替代预训练值。
        seed: 随机物品 embedding 的种子。
    """

    def __init__(
        self,
        model_root: Path,
        message: str,
        scratch_action_embedding: bool = False,
        seed: int = 2022,
    ) -> None:
        """解析产物路径并立即检查训练所需文件。

        Args:
            model_root: DeepFM 产物根目录。
            message: 文件名中的实验消息。
            scratch_action_embedding: 是否随机重置物品 embedding。
            seed: 随机重置使用的种子。

        Raises:
            FileNotFoundError: 预测矩阵或验证 embedding 缺失时抛出。
        """

        self.model_root = Path(model_root)
        self.message = message
        self.scratch_action_embedding = scratch_action_embedding
        self.seed = seed
        prefix = f"[{message}]"
        self.PREDICTION_MAT_PATH = str(
            self.model_root / "matsPre" / f"{prefix}_matPre.pickle"
        )
        self.VAR_MAT_PATH = str(
            self.model_root / "matsVar" / f"{prefix}_matVar.pickle"
        )
        self.Entropy_PATH = str(self.model_root / "entropy")
        self.USER_VAL_EMBEDDING_PATH = self.model_root / "embeddings" / (
            f"{prefix}_emb_user_val_M0.pt"
        )
        self.ITEM_VAL_EMBEDDING_PATH = self.model_root / "embeddings" / (
            f"{prefix}_emb_item_val_M0.pt"
        )
        self._validate_required_files()

    def _validate_required_files(self) -> None:
        """检查训练阶段真正需要的三个预计算产物。

        Raises:
            FileNotFoundError: 预测矩阵或验证集 embedding 缺失时抛出，并列出
                所有缺失路径。
        """

        required = (
            Path(self.PREDICTION_MAT_PATH),
            self.USER_VAL_EMBEDDING_PATH,
            self.ITEM_VAL_EMBEDDING_PATH,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            detail = "\n  - ".join(missing)
            raise FileNotFoundError(f"ROLeR 所需预计算产物缺失：\n  - {detail}")

    def eval(self, *args: Any, **kwargs: Any) -> "PrecomputedModelArtifacts":
        """兼容 ``BaseSimulatedEnv`` 的模型评估接口并返回自身。

        Args:
            *args: 为兼容模型接口而接受，当前不使用。
            **kwargs: 为兼容模型接口而接受，当前不使用。

        Returns:
            当前适配器实例。
        """

        del args, kwargs
        return self

    def load_val_user_item_embedding(
        self,
        model_i: int = 0,
        freeze_emb: bool = True,
    ) -> nn.ModuleDict:
        """加载验证用户与物品 embedding 并包装为统一模块字典。

        Args:
            model_i: ensemble 下标。轻量适配器只支持复现入口使用的第 0 个模型。
            freeze_emb: 是否冻结加载后的 embedding 权重。

        Returns:
            包含 ``feat_user`` 与 ``feat_item`` 的 ``ModuleDict``。

        Raises:
            ValueError: 请求第 0 个以外的模型时抛出。
        """

        if model_i != 0:
            raise ValueError("PrecomputedModelArtifacts 只提供 M0 验证 embedding。")
        user_embedding = _load_tensor(self.USER_VAL_EMBEDDING_PATH)
        item_embedding = _load_tensor(self.ITEM_VAL_EMBEDDING_PATH)
        if self.scratch_action_embedding:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            item_embedding = torch.empty_like(item_embedding)
            item_embedding.normal_(mean=0.0, std=0.01, generator=generator)
        return nn.ModuleDict(
            {
                "feat_user": nn.Embedding.from_pretrained(
                    user_embedding, freeze=freeze_emb
                ),
                "feat_item": nn.Embedding.from_pretrained(
                    item_embedding, freeze=freeze_emb
                ),
            }
        )


def _load_tensor(path: Path) -> torch.Tensor:
    """以 CPU 映射方式安全加载 embedding 张量。

    Args:
        path: PyTorch 张量文件。

    Returns:
        ``float32`` 二维张量。

    Raises:
        ValueError: 文件内容不是二维张量时抛出。
    """

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{path} 必须保存二维 torch.Tensor。")
    return value.detach().to(dtype=torch.float32)

