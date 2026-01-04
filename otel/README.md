# OpenTelemetry 本地服务启动指南

本项目使用 Docker 方式启动 OpenTelemetry Collector 服务，以解决运行时的连接错误。

## 问题说明

运行 `Fun_ASR_Nano/Inference_Fun_ASR_Nano_ONNX.py` 时可能会看到以下错误：
```
ConnectionRefusedError: [Errno 111] Connection refused
HTTPConnectionPool(host='localhost', port=4318)
```

这是因为 OpenTelemetry SDK 尝试将 metrics 发送到 `localhost:4318`，但该服务未运行。

## 解决方案：使用 Docker 启动 OpenTelemetry Collector

### 前置要求

- 已安装 Docker
- Docker 服务正在运行

### 快速开始

```bash
# 进入 otel 目录
cd otel

# 启动 OpenTelemetry Collector 服务
./start_otel_collector_docker.sh

# 现在可以运行推理脚本，不会再有连接错误
cd ../Fun_ASR_Nano
python3 Inference_Fun_ASR_Nano_ONNX.py

# 停止服务
cd ../otel
./stop_otel_collector_docker.sh
```

### 详细说明

#### 启动服务

```bash
./start_otel_collector_docker.sh
```

这个脚本会：
1. 检查 Docker 是否安装
2. 检查配置文件是否存在
3. 停止并删除已存在的容器（如果有）
4. 启动新的 OpenTelemetry Collector 容器
5. 监听端口 4318 (HTTP) 和 4317 (gRPC)

#### 停止服务

```bash
./stop_otel_collector_docker.sh
```

或者直接使用 Docker 命令：
```bash
docker stop otel-collector
docker rm otel-collector
```

#### 查看日志

```bash
docker logs -f otel-collector
```

#### 检查服务状态

```bash
# 检查容器是否运行
docker ps | grep otel-collector

# 检查端口是否监听
netstat -tuln | grep 4318
# 或
lsof -i :4318
```

## 配置文件说明

`otel-collector-config.yaml` 是 OpenTelemetry Collector 的配置文件，包含：

- **Receivers**: 接收器配置，监听端口 4318 (HTTP) 和 4317 (gRPC)
- **Processors**: 处理器配置，包括批处理和内存限制
- **Exporters**: 导出器配置，将数据输出到控制台（debug exporter）
- **Service**: 服务管道配置

## 备选方案：禁用 OpenTelemetry

如果不需要 OpenTelemetry 功能，可以在脚本中禁用：

在 `Fun_ASR_Nano/Inference_Fun_ASR_Nano_ONNX.py` 开头已添加：
```python
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['OTEL_METRICS_EXPORTER'] = 'none'
```

这样就不会尝试连接 OpenTelemetry 服务了。

## 故障排除

### 容器启动失败

1. 检查 Docker 是否运行：
   ```bash
   docker ps
   ```

2. 查看详细错误日志：
   ```bash
   docker logs otel-collector
   ```

3. 检查配置文件语法：
   ```bash
   docker run --rm -v $(pwd)/otel-collector-config.yaml:/etc/otelcol/config.yaml otel/opentelemetry-collector:latest --config=/etc/otelcol/config.yaml --dry-run
   ```

### 端口已被占用

如果端口 4318 已被占用，可以修改配置文件中的端口号，或停止占用端口的服务。

### 权限问题

确保脚本有执行权限：
```bash
chmod +x start_otel_collector_docker.sh stop_otel_collector_docker.sh
```

