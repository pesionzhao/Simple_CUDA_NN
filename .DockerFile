# 使用 PyTorch 基础镜像
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

# 设置工作目录
WORKDIR /workspace

# 安装常用工具和软件包
RUN apt-get update && apt-get install -y \
    build-essential\
    git \
    cmake \
    zsh \
    gdb \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Jupyter Notebook
RUN pip install --no-cache-dir jupyter

# 设置 zsh 为默认 shell
RUN chsh -s $(which zsh)

# 安装 Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# 安装一些 Jupyter 相关的扩展（可选）
RUN pip install --no-cache-dir notebook ipywidgets

# 默认命令：进入容器后，您可以手动启动 Jupyter
CMD ["zsh"]
