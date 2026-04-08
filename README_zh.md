# LLM API Prism

<p align="right">
  <a href="README.md">English</a> | 中文
</p>

N×N 多协议 LLM API 代理，支持任意两个 LLM 协议之间的无缝转换。客户端只需修改 `base_url`，其他代码完全不动。

## 架构设计

项目采用清晰的三层架构：

```
┌─────────────────────────────────────────────────────────────┐
│  第一层：路由层 (proxy.py)                                    │
│  - 自动生成 N×N 路由矩阵                                      │
│  - 组装转换链：输入协议 → IR → 后端协议                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第二层：链路层 (BackendAdapter)                               │
│  - 纯 HTTP 客户端                                             │
│  - 负责认证、请求头、流式传输                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第三层：转换层 (ProtocolConverter)                            │
│  - 纯协议 ↔ IR 双向数据转换                                    │
│  - IR 采用 OpenAI 格式作为统一中间表示                           │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

以 Anthropic 客户端 → DashScope 后端为例：

```
客户端请求 (Anthropic 格式)
    ↓
AnthropicConverter.request_to_ir()    →  IR (OpenAI 格式)
    ↓
DashScopeAdapter.chat_completion()    →  调用 DashScope API
    ↓
DashScopeConverter.response_to_ir()   →  IR (OpenAI 格式)
    ↓
AnthropicConverter.ir_to_response()   →  Anthropic 格式
    ↓
客户端响应 (Anthropic 格式)
```

## 支持的协议

| 协议 | 提供商 | 状态 |
|------|--------|------|
| OpenAI | OpenAI | ✅ 已支持 |
| Anthropic | Claude | ✅ 已支持 |
| Gemini | Google | ✅ 已支持 |
| DashScope | 阿里通义千问 | ✅ 已支持 |

4 个协议自动生成 4×4 = 16 条路由，覆盖所有协议组合。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python -m app.main
```

服务将在 `http://localhost:9876` 启动。

## 使用方式

### URL 格式

```
http://localhost:9876/{输入协议}/{后端协议}/{原始路径}
```

- **输入协议**：客户端使用的协议（openai / anthropic / gemini / dashscope）
- **后端协议**：实际调用的 LLM 服务（openai / anthropic / gemini / dashscope）
- **原始路径**：该协议的原生 API 路径

客户端只需将 `base_url` 改为 `http://localhost:9876/{输入协议}/{后端协议}`，其他代码完全不动。

### 示例 1：OpenAI 客户端 → DashScope 后端

```bash
curl http://localhost:9876/openai/dashscope/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_DASHSCOPE_API_KEY" \
  -d '{
    "model": "qwen-turbo",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 示例 2：Anthropic 客户端 → OpenAI 后端

```bash
curl http://localhost:9876/anthropic/openai/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 示例 3：Gemini 客户端 → Anthropic 后端

```bash
curl http://localhost:9876/gemini/anthropic/v1beta/models/gemini-pro:generateContent \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: YOUR_ANTHROPIC_API_KEY" \
  -d '{
    "contents": [{"parts": [{"text": "你好"}]}]
  }'
```

### 示例 4：DashScope 客户端 → Gemini 后端

```bash
curl http://localhost:9876/dashscope/gemini/api/v1/services/aigc/text-generation/generation \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GEMINI_API_KEY" \
  -d '{
    "model": "gemini-pro",
    "input": {
      "messages": [{"role": "user", "content": "你好"}]
    }
  }'
```

### 流式请求

所有协议均支持流式请求，触发方式与原生协议完全一致：

- **OpenAI / Anthropic**：请求体中设置 `"stream": true`
- **DashScope**：请求体中设置 `parameters.incremental_output: true`
- **Gemini**：使用 `streamGenerateContent` 路径（而非 `generateContent`）

```bash
# OpenAI 流式示例
curl http://localhost:9876/openai/dashscope/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen-turbo",
    "messages": [{"role": "user", "content": "讲个故事"}],
    "stream": true
  }'

# Gemini 流式示例（通过 URL 路径触发）
curl http://localhost:9876/gemini/openai/v1beta/models/gemini-pro:streamGenerateContent \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: YOUR_API_KEY" \
  -d '{
    "contents": [{"parts": [{"text": "讲个故事"}]}]
  }'
```

## 项目结构

```
app/
├── main.py                          # FastAPI 入口
├── models/
│   ├── common_models.py             # 通用 IR 模型（OpenAI 格式）
│   ├── openai_models.py             # OpenAI 专用模型
│   ├── anthropic_models.py          # Anthropic 专用模型
│   ├── gemini_models.py             # Gemini 专用模型
│   └── dashscope_models.py          # DashScope 专用模型
├── converters/
│   ├── base.py                      # ProtocolConverter 抽象基类
│   ├── openai_converter.py          # OpenAI ↔ IR 转换器
│   ├── anthropic_converter.py       # Anthropic ↔ IR 转换器
│   ├── gemini_converter.py          # Gemini ↔ IR 转换器
│   └── dashscope_converter.py       # DashScope ↔ IR 转换器
├── adapters/
│   ├── base.py                      # BackendAdapter 抽象基类
│   ├── openai_adapter.py            # OpenAI HTTP 客户端
│   ├── anthropic_adapter.py         # Anthropic HTTP 客户端
│   ├── gemini_adapter.py            # Gemini HTTP 客户端
│   └── dashscope_adapter.py         # DashScope HTTP 客户端
└── routers/
    └── proxy.py                     # N×N 路由矩阵生成
```

## 扩展新协议

新增协议只需 3 步，路由矩阵自动扩展：

### 第 1 步：定义模型

在 `app/models/{protocol}_models.py` 中定义协议的请求和响应数据模型。

### 第 2 步：实现转换器

在 `app/converters/{protocol}_converter.py` 中继承 `ProtocolConverter`，实现 7 个方法：

```python
from app.converters.base import ProtocolConverter

class NewProtocolConverter(ProtocolConverter):
    def request_to_ir(self, raw_body):
        """协议请求 → IR"""

    def ir_to_request(self, request, streaming=False):
        """IR → 协议请求体"""

    def response_to_ir(self, raw_response, model):
        """协议响应 → IR"""

    def ir_to_response(self, response):
        """IR → 协议响应"""

    def stream_line_to_ir(self, line, model, **kwargs):
        """协议 SSE 行 → IR 流式块"""

    def ir_to_stream_chunk(self, chunk):
        """IR 流式块 → 协议格式"""

    def stream_done_signal(self):
        """流结束信号"""
```

### 第 3 步：实现适配器

在 `app/adapters/{protocol}_adapter.py` 中继承 `BackendAdapter`，实现 2 个方法：

```python
from app.adapters.base import BackendAdapter

class NewProtocolAdapter(BackendAdapter):
    async def chat_completion(self, request):
        """非流式 HTTP 调用"""

    async def chat_completion_stream(self, request):
        """流式 HTTP 调用"""
```

完成后，在 `proxy.py` 的 `CONVERTER_REGISTRY` 和 `BACKEND_REGISTRY` 中注册即可，路由矩阵自动从 N×N 扩展为 (N+1)×(N+1)。

## 许可证

MIT
