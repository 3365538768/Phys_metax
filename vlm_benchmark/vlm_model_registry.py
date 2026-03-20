from dataclasses import dataclass, field
from typing import Dict, Type, Any

try:
    # 作为包运行: python -m vlm_benchmark.run_vlm_benchmark
    from vlm_benchmark.vlm_client_dashscope_mm import DashscopeMultiModalClient  # type: ignore
    from vlm_benchmark.vlm_client_openai import OpenAIVideoClient  # type: ignore
    from vlm_benchmark.vlm_client_seed20_ark import Seed20ArkClient  # type: ignore
except ModuleNotFoundError:
    # 作为脚本运行
    from vlm_client_dashscope_mm import DashscopeMultiModalClient  # type: ignore
    from vlm_client_openai import OpenAIVideoClient  # type: ignore
    from vlm_client_seed20_ark import Seed20ArkClient  # type: ignore


@dataclass
class VLMConfig:
    """
    单个 VLM 后端的配置：包括客户端类、默认模型名、base_url 和 API Key 环境变量名。
    如需长期维护/新增模型，只需要在这里增删条目即可。
    """

    client_cls: Type[Any]
    model: str
    base_url: str
    api_key_env: str
    client_kwargs: Dict[str, Any] = field(default_factory=dict)


VLM_REGISTRY: Dict[str, VLMConfig] = {
    # Qwen3.5-plus（官方 DashScope MultiModalConversation 视频抽帧方案）
    "qwen3.5-plus": VLMConfig(
        client_cls=DashscopeMultiModalClient,
        model="qwen3.5-plus",
        # 地域 base_url：国内（北京）/ 新加坡 / 弗吉尼亚，按需替换
        # 国内（北京）：https://dashscope.aliyuncs.com/api/v1
        # 新加坡：      https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        # 弗吉尼亚：    https://dashscope-us.aliyuncs.com/api/v1
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        client_kwargs={"fps": 5},
    ),
    # Seed2.0-VL（Ark 官方 AsyncArk /responses 视频调用方式）
    "seed2.0-vl": VLMConfig(
        client_cls=Seed20ArkClient,
        model="doubao-seed-2-0-lite-260215",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env="ARK_API_KEY",
        client_kwargs={"fps": 5},
    ),
    # GPT-5.4（OpenAI API，通过视频抽帧 + Vision 能力分析）
    # 使用 Responses API 可获取 reasoning.summary 思维过程，写入 output/<tag>/reasoning/
    "gpt5.4": VLMConfig(
        client_cls=OpenAIVideoClient,
        model="gpt-5.4",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        client_kwargs={
            "fps": 5,
            "max_frames": 20,
            "use_responses_api": True,
            "use_reasoning_summary": True,
        },
    ),
}


def create_vlm_client(tag: str):
    """
    根据标签创建对应的 VLM 客户端实例。
    tag 必须是 VLM_REGISTRY 的 key，例如：
      - "qwen2.5-vl"
      - "seed1.5-vl"
    """
    if tag not in VLM_REGISTRY:
        raise KeyError(f"未知的 VLM 标签: {tag}，可选值: {list(VLM_REGISTRY.keys())}")
    cfg = VLM_REGISTRY[tag]
    return cfg.client_cls(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key_env=cfg.api_key_env,
        **cfg.client_kwargs,
    )


__all__ = ["VLMConfig", "VLM_REGISTRY", "create_vlm_client"]

