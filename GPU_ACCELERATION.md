# 🚀 GPU加速说明

## 当前支持的加速方式

### 1. **CUDA加速（推荐）**
```bash
# 安装CUDA版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 使用GPU加速
python depth_video_converter.py input.mp4 output.mp4 --device cuda
```

### 2. **Web界面GPU加速**
```bash
# 启动时自动检测GPU
python gradio_app.py
# 在界面中选择 "cuda" 设备
```

## TensorRT加速（高级用户）

### 安装要求
```bash
# 1. 安装CUDA Toolkit (11.8+)
# 2. 安装cuDNN
# 3. 安装TensorRT SDK
# 4. 安装TensorRT Python包
pip install tensorrt
```

### 使用限制
- 需要NVIDIA GPU
- 模型需要转换为TensorRT格式
- 输入尺寸需要固定
- 配置复杂，适合生产环境

## 实际加速效果

### CUDA vs CPU
- **AI深度估计**: GPU比CPU快5-10倍
- **颜色深度计算**: GPU比CPU快2-3倍
- **视频处理**: GPU比CPU快3-5倍

### 内存使用
- **CPU模式**: 2-4GB RAM
- **GPU模式**: 4-8GB VRAM + 2-4GB RAM

## 推荐配置

### 入门用户
```bash
# 使用CPU，稳定可靠
python depth_video_converter.py input.mp4 output.mp4 --device cpu
```

### 有GPU用户
```bash
# 使用GPU，速度更快
python depth_video_converter.py input.mp4 output.mp4 --device cuda
```

### 高级用户
- 考虑TensorRT优化
- 使用ONNX Runtime
- 自定义模型量化

## 故障排除

### GPU不可用
```bash
# 检查CUDA是否安装
python -c "import torch; print(torch.cuda.is_available())"

# 检查GPU信息
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 内存不足
```bash
# 使用CPU模式
--device cpu

# 减少处理帧数
--max-frames 100

# 使用更小的模型
--model facebook/dpt-dinov2-small-kitti
```

## 性能优化建议

1. **选择合适的模型**
   - 高精度：Intel/dpt-large
   - 平衡：Intel/dpt-hybrid-midas  
   - 高速度：facebook/dpt-dinov2-small-kitti

2. **调整处理参数**
   - 使用depth_only格式（最快）
   - 限制处理帧数
   - 选择合适的输出分辨率

3. **硬件优化**
   - 使用SSD存储
   - 增加系统内存
   - 使用高性能GPU
