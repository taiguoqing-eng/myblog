---
title: "debian13编译安装Tdengine"
date: 2026-08-01T15:00:00+08:00
draft: false
description: "本文介绍如何在新版debian13上安装Tdengine数据库3.3.4.11"

# 分类（建议：大方向，1个即可）
categories:
  - "数据库"

# 标签（建议：具体技术点/工具，可以写多个）
tags:
  - "Tdengine"
  - "运维"
summary: "在 Debian 13（trixie）上从源码编译安装 TDengine 3.3.4.11：切换阿里源、安装编译依赖、cmake 构建、配置库路径并注册 systemd 开机自启，全程命令可直接复制。"
cover:
  image: "/images/covers/debian-tdengine.png"
  alt: "Debian13 编译安装 TDengine 封面"
  caption: ""
  relative: false
---

检查用户权限

usermod -aG sudo tgq
exit
newgrp sudo

### 修改为阿里源
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak

sudo vi /etc/apt/sources.list

deb https://mirrors.aliyun.com/debian/ trixie main contrib non-free non-free-firmware
deb https://mirrors.aliyun.com/debian/ trixie-updates main contrib non-free non-free-firmware
deb https://mirrors.aliyun.com/debian/ trixie-backports main contrib non-free non-free-firmware
deb https://mirrors.aliyun.com/debian-security/ trixie-security main contrib non-free non-free-firmware

备用：sudo bash -c 'echo "deb https://mirrors.aliyun.com/debian/ trixie main contrib non-free non-free-firmware
deb https://mirrors.aliyun.com/debian/ trixie-updates main contrib non-free non-free-firmware
deb https://mirrors.aliyun.com/debian/ trixie-backports main contrib non-free non-free-firmware
deb https://mirrors.aliyun.com/debian-security/ trixie-security main contrib non-free non-free-firmware" > /etc/apt/sources.list'
### 环境准备与依赖安装

sudo apt update
sudo apt install -y gcc cmake build-essential git libssl-dev
备用sudo apt install -y libjansson-dev libsnappy-dev liblzma-dev libz-dev pkg-config

tar -zxvf TDengine-ver-3.3.4.11.tar.gz
cd TDengine-ver-3.3.4.11
mkdir debug && cd debug
cmake .. && cmake --build . -j $(nproc)
sudo make install

### 配置库路径
echo "/usr/local/taos/driver" | sudo tee /etc/ld.so.conf.d/taos.conf
sudo ldconfig

### 启动服务
sudo systemctl start taosd
sudo systemctl enable taosd  # 设置开机自启
### 验证安装
taos
SHOW DATABASES;
USE test_db
SELECT * FROM meters;
