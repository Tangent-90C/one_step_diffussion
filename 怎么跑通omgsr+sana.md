这里用uv做python包管理器

安装uv
```
pip install uv
```

用uv创建虚拟环境

```
# 如果你的CUDA支持到CUDA 13.0
uv sync --extra cu130
# 如果你的CUDA支持到CUDA 12.8
uv sync --extra cu128
# 如果你只有CPU
uv sync --extra cpu
```

用uv的虚拟环境运行代码

```
uv run bash train_omgsr_sana_1024.sh
```