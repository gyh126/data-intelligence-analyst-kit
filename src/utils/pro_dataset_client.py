"""
专业数据集 Python 客户端
让 Jupyter Notebook 等独立 Python 环境能直接调用火山引擎「专业数据集」API，
无需依赖 TRAE MCP 运行时。

MCP 端点：https://datapro.hqd.cn-beijing.volces.com/mcp
鉴权：X-Hqd-Api-Key 请求头

使用方式：
    from utils.pro_dataset_client import ProDatasetClient

    client = ProDatasetClient(api_key="hqd_sk_xxx")
    result = client.search("比亚迪 近一年全国销量")
    print(result)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class ProDatasetResult:
    """专业数据集检索结果"""
    code: int
    msg: str
    dataset_type: str
    items: list
    raw: dict
    trace_id: str = ""

    @property
    def success(self) -> bool:
        return self.code == 0

    def to_dataframe(self) -> "pd.DataFrame":
        """尝试将 items 转为 DataFrame（适用于表格型数据）"""
        import pandas as pd

        if not self.items:
            return pd.DataFrame()
        # items 可能是 dict 列表或嵌套结构，尝试扁平化
        flat = []
        for item in self.items:
            if isinstance(item, dict):
                row = {}
                for k, v in item.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        row[k] = v
                    elif isinstance(v, list):
                        # 展开列表字段为多行
                        for sub in v:
                            if isinstance(sub, dict):
                                sub_row = {**row, **sub}
                                flat.append(sub_row)
                        continue
                    else:
                        row[k] = str(v)
                else:
                    flat.append(row)
        return pd.DataFrame(flat)


class ProDatasetClient:
    """
    专业数据集 MCP HTTP 客户端。

    通过 MCP Streamable HTTP 协议调用 dataPro_search 工具。
    """

    DEFAULT_URL = "https://datapro.hqd.cn-beijing.volces.com/mcp"

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: str = DEFAULT_URL,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("PRO_DATASET_API_KEY")
        if not self.api_key:
            raise ValueError(
                "缺少 API Key。请通过参数传入或设置环境变量 PRO_DATASET_API_KEY。\n"
                "获取方式：火山引擎控制台 → 专业数据集 → API Key"
            )
        self.url = url
        self.timeout = timeout
        self._session_id: Optional[str] = None
        self._initialized = False

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-Hqd-Api-Key": self.api_key,
            "X-Hqd-Extra-Info": "trae",
        }
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    def _rpc(self, method: str, params: Optional[dict] = None) -> dict:
        """发送 JSON-RPC 请求"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        resp = requests.post(
            self.url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
            stream=True,
        )

        # 响应可能是 SSE 或直接 JSON
        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse(resp)
        else:
            return resp.json()

    @staticmethod
    def _parse_sse(resp) -> dict:
        """解析 SSE 响应，提取最后一个 data 块"""
        last_data = None
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str:
                    try:
                        last_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        pass
        return last_data or {}

    def initialize(self) -> dict:
        """MCP 握手初始化"""
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "pro-dataset-python-client", "version": "0.1.0"},
            },
        )
        # 提取 session id
        if isinstance(result, dict):
            self._session_id = result.get("result", {}).get("sessionId")
        self._initialized = True
        # 发送 initialized 通知
        try:
            requests.post(
                self.url,
                headers=self._headers(),
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                timeout=self.timeout,
            )
        except Exception:
            pass
        return result

    def search(self, query: str) -> ProDatasetResult:
        """
        调用 dataPro_search 工具检索专业数据。

        Args:
            query: 自然语言查询，需包含目标主体和关键词。
                   示例："比亚迪 近一年全国销量趋势"、"贵州茅台 600519.SH 最近一周价格"

        Returns:
            ProDatasetResult 对象
        """
        if not self._initialized:
            self.initialize()

        result = self._rpc(
            "tools/call",
            {"name": "dataPro_search", "arguments": {"query": query}},
        )

        # 解析结果
        code = result.get("code", -1) if isinstance(result, dict) else -1
        msg = result.get("msg", "") if isinstance(result, dict) else ""
        items = result.get("items", []) if isinstance(result, dict) else []
        dataset_type = result.get("dataset_type", "") if isinstance(result, dict) else ""
        trace_id = result.get("trace_id", "") if isinstance(result, dict) else ""

        # 如果 result 嵌套在 result 字段中
        if "result" in result and isinstance(result["result"], dict):
            inner = result["result"]
            code = inner.get("code", code)
            msg = inner.get("msg", msg)
            items = inner.get("items", items)
            dataset_type = inner.get("dataset_type", dataset_type)
            trace_id = inner.get("trace_id", trace_id)

        return ProDatasetResult(
            code=code,
            msg=msg,
            dataset_type=dataset_type,
            items=items,
            raw=result,
            trace_id=trace_id,
        )

    def search_to_dataframe(self, query: str) -> "pd.DataFrame":
        """检索并尝试转为 DataFrame"""
        result = self.search(query)
        if not result.success:
            raise RuntimeError(f"检索失败: {result.msg}")
        return result.to_dataframe()


if __name__ == "__main__":
    # 测试（需设置 PRO_DATASET_API_KEY）
    key = os.getenv("PRO_DATASET_API_KEY")
    if not key:
        print("请设置环境变量 PRO_DATASET_API_KEY")
        exit(1)
    client = ProDatasetClient(api_key=key)
    r = client.search("比亚迪 近一年全国销量趋势")
    print(f"code={r.code}, msg={r.msg}, type={r.dataset_type}")
    print(f"items count: {len(r.items)}")
    if r.items:
        print(json.dumps(r.items[0], ensure_ascii=False, indent=2)[:500])
