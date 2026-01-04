#!/bin/bash
# 停止 Docker 容器中的 OpenTelemetry Collector 服务

CONTAINER_NAME="otel-collector"

if docker ps | grep -q "$CONTAINER_NAME"; then
    echo "正在停止 OpenTelemetry Collector 容器..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    echo "✓ OpenTelemetry Collector 容器已停止并删除"
elif docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo "容器已停止，正在删除..."
    docker rm $CONTAINER_NAME
    echo "✓ 容器已删除"
else
    echo "未找到运行中的 OpenTelemetry Collector 容器"
fi

