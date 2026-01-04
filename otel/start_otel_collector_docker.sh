#!/bin/bash
# 使用 Docker 启动 OpenTelemetry Collector 服务
# 如果系统没有安装 OpenTelemetry Collector，可以使用这个脚本

set -e

CONTAINER_NAME="otel-collector"
CONFIG_FILE="$(dirname "$0")/otel-collector-config.yaml"
PORT_HTTP=4318
PORT_GRPC=4317

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 检查容器是否已运行
if docker ps | grep -q "$CONTAINER_NAME"; then
    echo "OpenTelemetry Collector 容器已在运行"
    echo "停止现有容器..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

echo "启动 OpenTelemetry Collector (Docker)..."
echo "配置文件: $CONFIG_FILE"
echo "HTTP 端口: $PORT_HTTP"
echo "gRPC 端口: $PORT_GRPC"
echo ""

# 启动 Docker 容器
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT_HTTP:4318 \
    -p $PORT_GRPC:4317 \
    -v "$(pwd)/$CONFIG_FILE:/etc/otelcol/config.yaml" \
    otel/opentelemetry-collector:latest \
    --config=/etc/otelcol/config.yaml

echo "✓ OpenTelemetry Collector 容器已启动"
echo ""
echo "查看日志: docker logs -f $CONTAINER_NAME"
echo "停止服务: docker stop $CONTAINER_NAME"
echo "或运行: ./stop_otel_collector_docker.sh"

