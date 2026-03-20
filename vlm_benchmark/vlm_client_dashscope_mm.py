import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from vlm_benchmark.parse_utils import (
        extract_json_from_response,
        extract_reasoning_before_json,
    )
except ModuleNotFoundError:
    from parse_utils import (
        extract_json_from_response,
        extract_reasoning_before_json,
    )


class DashscopeMultiModalClient:
    """
    使用 DashScope 官方 SDK 的 MultiModalConversation 调用多模态模型（含视频）。

    参考文档用法：
      - dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
      - messages: [{'role':'user','content':[{'video': 'file:///abs/path.mp4', 'fps':2},{'text':'...'}]}]
      - MultiModalConversation.call(api_key=..., model=..., messages=...)
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        fps: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.fps = int(fps)

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置，无法调用 DashScope。")
        self.api_key = api_key

        try:
            import dashscope  # type: ignore
            from dashscope import MultiModalConversation  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 dashscope SDK。请先执行：pip install dashscope"
            ) from exc

        # 设置 DashScope API base url（地域）
        dashscope.base_http_api_url = self.base_url
        self._mm = MultiModalConversation

    def predict_video(
        self,
        video_path: str,
        system_prompt: str,
        user_prompt: str,
        timeout: Optional[float] = None,
        video_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"找不到视频文件: {video_path}")

        # DashScope 官方示例使用 file:// 形式。
        # 重要：路径中可能含 '+'（如科学计数法 e+04），部分 URI 解析会把 '+' 当空格。
        # 使用 Path.as_uri() 做 percent-encoding，确保 '+' -> %2B，避免 “file not exists” 误判。
        video_uri = Path(os.path.abspath(video_path)).resolve().as_uri()

        # 这里仍沿用我们 benchmark 的 prompt：system + user
        # MultiModalConversation 只需要 role=user 也能跑，但我们把 system prompt 合并进 text，避免丢失约束。
        merged_text = (system_prompt.strip() + "\n\n" + user_prompt.strip()).strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"video": video_uri, "fps": self.fps},
                    {"text": merged_text},
                ],
            }
        ]

        # timeout 参数：dashscope SDK 是否支持传入 timeout 取决于版本；
        # 这里不强行传，保持兼容性。
        # OSS 上传阶段偶发 TLS EOF / 网络抖动，做有限次重试
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                resp = self._mm.call(
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                )
                break
            except Exception as exc:  # dashscope 会封装 requests 异常，统一捕获
                last_exc = exc
                wait_s = min(2 ** (attempt - 1), 4)
                print(f"[WARN] DashScope 调用失败（第 {attempt}/3 次），{wait_s}s 后重试: {exc}")
                time.sleep(wait_s)
        else:
            raise last_exc  # type: ignore[misc]

        # 尝试取出 reasoning（若 SDK 返回）
        reasoning_out = ""
        try:
            content_items_dbg = resp.output.choices[0].message.content  # type: ignore[attr-defined]
            for it in content_items_dbg:
                if isinstance(it, dict) and ("reasoning_content" in it):
                    reasoning_out = str(it.get("reasoning_content") or "")
                    break
        except Exception:
            reasoning_out = ""

        # 取出模型文本输出
        # 文档示例：response.output.choices[0].message.content[0]["text"]
        try:
            content_items = resp.output.choices[0].message.content
            # 找到第一个含 text 的 item
            text_out = None
            for it in content_items:
                if isinstance(it, dict) and "text" in it:
                    text_out = it["text"]
                    break
            if text_out is None:
                text_out = str(content_items)
        except Exception as exc:
            raise RuntimeError(f"解析 DashScope 返回失败: {resp}") from exc

        raw_text = str(text_out).strip()
        parsed = extract_json_from_response(raw_text)

        if not isinstance(parsed, dict):
            raise ValueError(f"期望 JSON 对象，但得到: {type(parsed)}")

        parsed["__raw_text__"] = raw_text
        reasoning = reasoning_out or extract_reasoning_before_json(raw_text)
        if reasoning:
            parsed["__reasoning__"] = reasoning
        return parsed


__all__ = ["DashscopeMultiModalClient"]

