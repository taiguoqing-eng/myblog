---
title: "昇腾 310P 部署 GPUSTACK 集群"
date: 2026-07-01T12:00:00+08:00
draft: false
description: "使用 GPUStack 极限部署 DeepSeek-V4-Flash 的全新实操方案"

categories:
  - "AI大模型"

tags:
  - "AI"
  - "DEEPSEEK"
  - "GPUSTACK"
---

### GPUSTACK

GPUStack 是一个非常优秀的轻量级大模型多机整合与服务化框架，它支持异构算力和私有化集群。虽然它底层对华为昇腾（CANN）和鲲鹏（ARM64）的自动化原生支持可能不如传统 NVIDIA N卡那样“即插即用”，但通过其支持的 **vLLM（Ascend 分支）后端** 配合 Docker，我们可以非常优雅地实现多机并联。

以下是使用 GPUStack 极限部署 DeepSeek-V4-Flash 的全新实操方案：

---

### 🛠️ GPUStack 多机架构设计

由于 310P3 运行大模型必须依赖华为 CANN 环境和特定的 vLLM/MindIE 算子，在 GPUStack 体系下，我们推荐采用 **「Host 宿主机 GPUStack 管理 + Container 容器内算力节点」** 的架构。

* **节点 0 (Master 节点)**：运行 GPUStack Server（控制台与调度中心）+ 运行 GPUStack Worker（提供本地两块 310P3 算力）。
* **节点 1、2、3 (Worker 节点)**：仅运行 GPUStack Worker 注册到 Master，提供各自的 310P3 算力。

---

## 第一阶段：宿主机基础环境（4台机器同步执行）

无论用什么框架，底层的驱动和固件是绕不开的。请确保 4台 机器均已完成以下操作：

### 1. 安装昇腾 310P3 驱动与固件
参考上一篇文档，确保 `npu-smi info` 能够正常看到每台机器的 96GB 显存。

### 2. 安装 Docker 与 Ascend-Docker-Runtime
为了让 Docker 容器能够调用宿主机的 310P3 芯片，必须安装华为官方的容器运行时：

```bash
# 安装常规 Docker
sudo yum install -y docker-ce
sudo systemctl start docker && sudo systemctl enable docker

# 下载并安装华为 Ascend Docker Runtime
# 请去昇腾官网下载 Ascend-docker-runtime_*.run
chmod +x Ascend-docker-runtime_*.run
sudo ./Ascend-docker-runtime_*.run --install

# 重启 Docker 使运行时生效
sudo systemctl restart docker
```

## 第二阶段：部署 GPUStack Server（仅在 节点0 执行）
在主节点上，我们直接拉起 GPUStack 的控制中心。
```bash
curl -sfL [https://get.gpustack.ai](https://get.gpustack.ai) | sh -

# 启动成功后，查看初始登录 Token
cat /var/lib/gpustack/token
```
此时你可以通过浏览器访问 http://192.168.10.11:10114 登录 GPUStack WebUI。

## 第三阶段：配置昇腾专属 Worker 容器（4台机器同步执行）
由于 GPUStack 原生的二进制 Worker 无法直接编译昇腾复杂的底层通信（HCCL），我们需要采用 HuggingFace/vLLM-Ascend 或 华为官方 MindIE 的 Docker 镜像 作为 Worker 的运行载体。这里以社区广泛兼容的 vllm-ascend 镜像为例：
1. 拉取支持 310P3/910B 的 vLLM 镜像
```bash
docker pull flashel/vllm-ascend:latest # 或使用华为官方提供的 MindIE 镜像
```
2. 将容器作为 GPUStack Worker 启动并注册在 4 台机器上，将整个容器伪装成 Worker 注入到 节点0 的 Server 中。在 节点0 执行：
```bash
docker run -d --name gpustack-worker-0 \
  --net=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /data/models:/data/models \
  flashel/vllm-ascend:latest \
  gpustack worker --server-url [http://192.168.10.11:10114](http://192.168.10.11:10114) --token <你的SERVER_TOKEN>
```
在 节点1、2、3 执行：
只需将上面的命令复制到另外三台机器执行即可（确保 --server-url 依然指向 节点0）。
⚠️ 关键点解释：--device=/dev/davinci* 和挂载 driver 目录是为了把宿主机的 310P3 显卡硬直通进容器。通过 --net=host 让 4 个容器共享物理网络，以保证 MoE 模型的分布式 All-to-All 通信。

启动完成后，刷新 节点0 的 GPUStack WebUI，你会在 "Nodes" 页面看到 4 个健康的节点，每台机器都识别出了对应大显存的昇腾 NPU。

## 第四阶段：在 GPUStack 界面极限部署 DeepSeek-V4-Flash
现在，你不需要再去手写繁琐的 rank_table.json 和环境变量了，所有的分布式切分全部在 WebUI 页面通过可视化表单完成。

1. 添加模型 (Models -> Add Model)
Model Name: deepseek-v4-flash-int4

Source: 选择 Local Path（如果已经提前把量化好的模型文件放在了 4 台机器的 /data/models/deepseek-v4-flash-int4 目录下）。

2. 配置分布式推理参数（核心压榨配置）在模型的高级设置（Advanced Settings）中，按以下参数配置以适配 310P3 的硬件极限：

| **配置项**                         | **设定值**                                      | **作用说明**                              |
| ------------------------------- | -------------------------------------------- | ------------------------------------- |
| **Backend**                     | `vLLM`                                       | 强制指定使用容器内适配了昇腾的 vLLM 后端               |
| **Tensor Parallel Size (TP)**   | `2`                                          | 在单台机器内部的 2 个显卡芯片间做张量并行                |
| **Pipeline Parallel Size (PP)** | `4`                                          | 纵向跨 4 台服务器做流水线切分，规避物理网络带宽瓶颈           |
| **Max Model Length**            | `32768`                                      | 上下文先限制在 32K，防止首轮实验 OOM                |
| **Max Num Seq (Batch Size)**    | `1`                                          | **必须设为 1**，310P3 的算力无法承受多并发的 V4-Flash |
| **Additional Arguments**        | `--quantization awq` 或 `--quantization gptq` | 明确告诉后端这是 INT4 量化模型                    |


## 🔍 极限技术验证与监控
点击部署后，GPUStack 会自动在 4 台机器的容器之间拉起通信。你可以通过以下方式验证和压榨这套系统：

观察控制台日志：在 GPUStack 的 Model Logs 页面，你会看到 4 个节点通过网卡建立 HCCL（华为集合通信）握手的过程。

算力压榨监控：在 4 台宿主机上分别开一个终端运行：
```bash
watch -n 1 npu-smi info
```
当你通过 GPUStack 自带的 Playground 发送 Prompt 时，你会非常直观地看到：由于配置了 PP=4（流水线并行），4台机器的 310P3 显存占用是满的（约 36G+），但 AI Core 利用率（算力）并不是同时拉满，而是像接力赛一样，节点0 闪烁 -> 节点1 闪烁 -> 节点2 闪烁 -> 节点3 闪烁。

![img](/images/image-1.png)
![img](/images/image.png)


## 💡 结论
利用 GPUStack + Docker 容器直通，你免去了手动管理分布式多进程拉起的痛苦。虽然受限于 310P3 LPDDR4X 内存带宽，最终的吐字速度（Token/s）依然是一秒几个字的“极限验证速度”，但你用最现代化的私有大模型 management 框架，成功在老旧/魔改推理卡上完成了对 2840亿 参数顶级 MoE 模型的驯服！
