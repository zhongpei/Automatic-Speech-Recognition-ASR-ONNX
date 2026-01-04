# Automatic Speech Recognition ASR ONNX

Fun ASR Nano ONNX 导出和推理项目

## 项目结构

```
.
├── Fun_ASR_Nano/          # 主要代码目录
│   ├── Export_Fun_ASR_Nano.py          # ONNX 模型导出脚本
│   ├── Inference_Fun_ASR_Nano_ONNX.py  # ONNX 模型推理脚本
│   ├── Optimize_ONNX.py                # ONNX 模型优化脚本
│   ├── STFT_Process.py                  # STFT 处理模块
│   ├── models/                          # 模型文件目录
│   └── example/                         # 示例音频文件
├── otel/                    # OpenTelemetry 服务配置
│   ├── otel-collector-config.yaml
│   ├── start_otel_collector_docker.sh
│   ├── stop_otel_collector_docker.sh
│   └── README.md
├── pyproject.toml          # uv 项目配置
└── README.md              # 本文件
```

## 环境要求

- Python >= 3.8
- uv (Python 包管理器)

## 依赖管理

项目使用 `uv` 进行依赖管理。配置文件为 `pyproject.toml`。

### 关键依赖

- torch, torchaudio
- numpy
- onnxruntime
- pydub
- funasr
- transformers

### 安装依赖

**注意**: 由于 `funasr` 的某些依赖（如 `llvmlite`）与 Python 3.10+ 存在兼容性问题，如果使用 `uv sync` 安装失败，可以使用系统 Python 环境：

```bash
# 使用系统 Python 安装依赖
pip install torch torchaudio numpy onnxruntime pydub funasr transformers
```

或者使用 Python 3.9 环境：

```bash
# 使用 Python 3.9 创建虚拟环境
uv venv --python 3.9
uv sync
```

## 使用方法

### 1. 导出 ONNX 模型

```bash
cd Fun_ASR_Nano
python3 Export_Fun_ASR_Nano.py
```

导出的模型将保存在 `./models_onnx/` 目录中。

### 2. 运行推理

```bash
cd Fun_ASR_Nano
python3 Inference_Fun_ASR_Nano_ONNX.py
```

### 3. 启动 OpenTelemetry 服务（可选）

如果遇到 OpenTelemetry 连接错误，可以启动本地服务：

```bash
cd otel
./start_otel_collector_docker.sh
```

详细说明请参考 `otel/README.md`。

## 配置说明

### 模型路径

- 模型文件: `./models/`
- Tokenizer: `./models/Qwen3-0.6B/`
- ONNX 输出: `./models_onnx/`

### 导出配置

在 `Export_Fun_ASR_Nano.py` 中：
- `OPSET = 13` - ONNX opset 版本（已修复兼容性问题）

## 故障排除

### OpenTelemetry 连接错误

如果看到 `ConnectionRefusedError`，可以：
1. 启动 OpenTelemetry 服务（见上方）
2. 或在脚本中禁用 OpenTelemetry（已在 `Inference_Fun_ASR_Nano_ONNX.py` 中配置）

### 依赖安装问题

如果 `uv sync` 失败，使用系统 Python 环境或 Python 3.9。

## 许可证

请参考项目根目录的许可证文件。
