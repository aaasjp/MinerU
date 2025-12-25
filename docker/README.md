# MinerU Docker 部署指南

本文档介绍如何使用 Docker 部署 MinerU 服务，包括使用 Docker Compose 和直接使用 Dockerfile 两种方式。

## 目录结构

```
docker/
├── README.md           # 本文档
├── compose.yaml        # Docker Compose 配置文件
├── global/             # 全球区域 Dockerfile
│   └── Dockerfile
└── china/              # 中国区域 Dockerfile（使用国内镜像源）
    ├── Dockerfile
    ├── maca.Dockerfile
    ├── npu.Dockerfile
    └── ppu.Dockerfile
```

## 文件说明

### Dockerfile

Dockerfile 用于构建 MinerU 的 Docker 镜像，主要特点：

- **基础镜像**：使用 `vllm/vllm-openai` 作为基础镜像，默认集成了 vllm 推理加速框架
- **模型下载**：构建时自动下载所需模型
- **ENTRYPOINT**：设置了统一的入口点，自动设置 `MINERU_MODEL_SOURCE=local` 环境变量

#### ENTRYPOINT 说明

```dockerfile
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]
```

这个 ENTRYPOINT 的作用：

1. **设置环境变量**：`export MINERU_MODEL_SOURCE=local` - 确保容器内使用本地模型源
2. **执行传入命令**：`exec "$@"` - 执行所有传入的参数作为命令
3. **进程替换**：使用 `exec` 替换当前进程，使被执行的命令成为容器主进程（PID 1），确保信号能正确传递
4. **`--` 的作用**：告诉 bash 后续参数是位置参数而不是选项

### compose.yaml

`compose.yaml` 定义了三个独立的 MinerU 服务，通过 `profiles` 机制可以按需启动：

1. **mineru-openai-server**：OpenAI 兼容服务器（端口 30000）
2. **mineru-api**：REST API 服务（端口 8000）
3. **mineru-gradio**：Gradio Web UI 服务（端口 7860）

## 构建 Docker 镜像

### 使用全球镜像源（推荐海外用户）

源码方式安装
```bash
cd docker
docker build -t mineru:latest -f Dockerfile.sourcecode .
```

```bash
cd docker
docker build -t mineru:latest -f global/Dockerfile .
```

### 使用中国镜像源（推荐国内用户）

```bash
cd docker
docker build -t mineru:latest -f china/Dockerfile .
```

> [!TIP]
> - `global/Dockerfile` 使用 HuggingFace 作为模型源
> - `china/Dockerfile` 使用 ModelScope 作为模型源，并使用国内镜像加速下载
> - 如果您的 GPU 是 Turing 架构或更早（Compute Capability < 8.0），需要修改 Dockerfile 中的基础镜像为 `vllm/vllm-openai:v0.10.2`

## 启动服务

### 方式一：使用 Docker Compose（推荐）

Docker Compose 提供了更便捷的服务管理方式，支持通过 `profiles` 选择性启动服务。

#### 启动 OpenAI 兼容服务器

```bash
docker compose -f compose.yaml --profile openai-server up -d
```

- 服务端口：`30000`
- 访问地址：`http://<server_ip>:30000`
- 用途：提供 OpenAI 兼容的 API 接口，可通过 `vlm-http-client` 后端连接

#### 启动 Web API 服务

```bash
docker compose -f compose.yaml --profile api up -d
```

- 服务端口：`8000`
- API 文档：`http://<server_ip>:8000/docs`
- 主要端点：`POST /file_parse` - 解析 PDF/图片文件

#### 启动 Gradio WebUI 服务

```bash
docker compose -f compose.yaml --profile gradio up -d
```

- 服务端口：`7860`
- 访问地址：`http://<server_ip>:7860`
- 特性：启用 vllm 引擎，支持 API 访问

#### 查看服务状态

```bash
# 查看运行中的容器
docker compose -f compose.yaml ps

# 查看日志
docker compose -f compose.yaml logs -f

# 查看特定服务的日志
docker logs mineru-api -f
```

#### 停止服务

```bash
# 停止特定 profile 的服务
docker compose -f compose.yaml --profile api down
docker compose -f compose.yaml --profile openai-server down
docker compose -f compose.yaml --profile gradio down
```

### 方式二：直接使用 Dockerfile

如果不想使用 Docker Compose，可以直接使用 `docker run` 命令启动服务。

#### 启动 API 服务

**前台运行：**

```bash
docker run --gpus all \
  --shm-size 32g \
  -p 8000:8000 \
  --ipc=host \
  -e MINERU_MODEL_SOURCE=local \
  mineru:latest \
  mineru-api --host 0.0.0.0 --port 8000
```

**后台运行（推荐）：**

```bash
docker run -d --name mineru-api \
  --gpus all \
  --shm-size 32g \
  -p 8002:8000 \
  --ipc=host \
  -e MINERU_MODEL_SOURCE=local \
  crpi-14w4py2o64ex3hc2.cn-qingdao.personal.cr.aliyuncs.com/songjianping2022/mineru:sourcecode \
  mineru-api --host 0.0.0.0 --port 8000
```

**挂载数据卷（持久化输出）：**

```bash
docker run -d --name mineru-api \
  --gpus all \
  --shm-size 32g \
  -p 8000:8000 \
  --ipc=host \
  -e MINERU_MODEL_SOURCE=local \
  -v $(pwd)/output:/app/output \
  mineru:latest \
  mineru-api --host 0.0.0.0 --port 8000
```

#### 启动 OpenAI 兼容服务器

```bash
docker run -d --name mineru-openai-server \
  --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  --ipc=host \
  -e MINERU_MODEL_SOURCE=local \
  mineru:latest \
  mineru-openai-server --engine vllm --host 0.0.0.0 --port 30000
```

#### 启动 Gradio WebUI

```bash
docker run -d --name mineru-gradio \
  --gpus all \
  --shm-size 32g \
  -p 7860:7860 \
  --ipc=host \
  -e MINERU_MODEL_SOURCE=local \
  mineru:latest \
  mineru-gradio --server-name 0.0.0.0 --server-port 7860 --enable-vllm-engine true
```

#### 管理容器

```bash
# 查看容器日志
docker logs mineru-api -f

# 查看容器状态
docker ps | grep mineru

# 停止容器
docker stop mineru-api

# 删除容器
docker rm mineru-api
```

## 执行流程说明

### 使用 Docker Compose 启动 API 服务

当执行 `docker compose --profile api up -d` 时：

1. Docker Compose 读取 `compose.yaml` 文件
2. 找到 `profiles: ["api"]` 的服务配置
3. 启动容器，执行顺序：
   - **Dockerfile 的 ENTRYPOINT**：`export MINERU_MODEL_SOURCE=local && exec "$@"`
   - **compose.yaml 的 entrypoint**：`mineru-api`
   - **compose.yaml 的 command**：`--host 0.0.0.0 --port 8000`
4. 最终执行：`mineru-api --host 0.0.0.0 --port 8000`
5. 启动 FastAPI 服务，监听 `0.0.0.0:8000`

### 直接使用 Dockerfile 启动

当执行 `docker run ... mineru-api --host 0.0.0.0 --port 8000` 时：

1. Dockerfile 的 ENTRYPOINT 执行：`export MINERU_MODEL_SOURCE=local && exec "$@"`
2. `"$@"` 接收的参数是：`mineru-api --host 0.0.0.0 --port 8000`
3. 最终执行：`mineru-api --host 0.0.0.0 --port 8000`
4. 启动 FastAPI 服务，监听 `0.0.0.0:8000`

## 参数说明

### Docker Run 常用参数

- `--gpus all`：启用所有 GPU（需要 NVIDIA Docker 支持）
- `--shm-size 32g`：设置共享内存大小（vllm 可能需要较大共享内存）
- `-p 8000:8000`：端口映射（宿主机端口:容器端口）
- `--ipc=host`：使用主机 IPC 命名空间（提升性能）
- `-e MINERU_MODEL_SOURCE=local`：设置环境变量
- `-v $(pwd)/output:/app/output`：挂载数据卷（持久化存储）
- `-d`：后台运行（detached mode）

### compose.yaml 配置说明

- **profiles**：服务分组，可以按需启动特定服务
- **ulimits**：资源限制配置
- **deploy.resources.reservations.devices**：GPU 设备配置
  - `device_ids: ["0"]`：使用 GPU 0，多 GPU 可配置为 `["0", "1"]`
- **healthcheck**：健康检查配置（仅 openai-server 服务）

## 服务访问

### API 服务

- **API 文档**：`http://localhost:8000/docs`
- **主要端点**：`POST http://localhost:8000/file_parse`
- **功能**：解析 PDF/图片文件，支持多种后端（pipeline、vlm-transformers、vlm-vllm-async-engine 等）

### OpenAI 兼容服务器

- **访问地址**：`http://localhost:30000`
- **用途**：提供 OpenAI 兼容接口
- **客户端连接示例**：
  ```bash
  mineru -p <input_path> -o <output_path> -b vlm-http-client -u http://<server_ip>:30000
  ```

### Gradio WebUI

- **访问地址**：`http://localhost:7860`
- **功能**：可视化 Web 界面，支持文件上传和解析

## 注意事项

1. **GPU 要求**：
   - 需要 Volta 架构或更新的 GPU（Compute Capability >= 7.0）
   - 可用显存 >= 8GB
   - 显卡驱动支持 CUDA 12.8 或更高版本

2. **vllm 限制**：
   - 由于 vllm 会预分配显存，通常不能同时运行多个使用 vllm 的服务
   - 启动新服务前，确保其他可能使用显存的服务已停止

3. **多 GPU 配置**：
   - 在 `compose.yaml` 中修改 `device_ids` 为 `["0", "1"]` 使用多 GPU
   - 可以添加 `--data-parallel-size 2` 参数启用 vllm 的多 GPU 并行模式

4. **显存不足**：
   - 如果遇到 VRAM 不足，可以降低 `--gpu-memory-utilization` 参数（如 `0.5` 或 `0.4`）

5. **环境变量**：
   - `MINERU_MODEL_SOURCE=local` 已在 Dockerfile 的 ENTRYPOINT 中设置
   - 如需覆盖，可在 `docker run` 或 `compose.yaml` 中显式指定

## 故障排查

### 查看容器日志

```bash
# 查看所有服务日志
docker compose -f compose.yaml logs -f

# 查看特定服务日志
docker logs mineru-api -f
```

### 检查 GPU 访问

```bash
# 进入容器检查 GPU
docker exec -it mineru-api nvidia-smi
```

### 检查端口占用

```bash
# 检查端口是否被占用
netstat -tuln | grep 8000
```

### 重启服务

```bash
# 重启特定服务
docker compose -f compose.yaml restart mineru-api

# 或使用 docker 命令
docker restart mineru-api
```

## 相关文档

- [MinerU 快速开始](../docs/zh/quick_start/index.md)
- [Docker 部署文档](../docs/zh/quick_start/docker_deployment.md)
- [API 使用说明](../docs/zh/usage/quick_usage.md)

