# 🎥 深度视频转换器

一个功能强大的深度视频转换程序，支持AI深度估计和创新的颜色深度计算，可以将普通视频/图像转换为包含深度信息的视频/图像。

## ✨ 核心功能

### 🧠 AI深度估计
- 使用先进的深度学习模型（DPT系列）进行深度估计
- 支持多种预训练模型，平衡精度和速度
- GPU加速处理，大幅提升处理速度

### 🎨 颜色深度计算（新功能）
- **保持黑色区域不变** - 黑色背景深度始终为0
- **主色调深度增强** - 主色调偏多的区域获得更高深度值
- **自定义颜色映射** - 支持为特定颜色指定深度值
- **智能颜色分析** - 使用K-means聚类分析图像颜色分布

### 🌐 Web界面
- 基于Gradio的现代化Web界面
- 实时参数调整和预览
- 响应式设计，支持各种设备

## 🚀 快速开始

### 安装依赖

```bash
# 使用requirements.txt安装
pip install -r requirements.txt

# 或手动安装核心依赖
pip install torch torchvision opencv-python transformers scikit-learn gradio tqdm numpy pillow
```

### 基本使用

#### 命令行方式
```bash
# AI深度估计
python depth_video_converter.py input.mp4 output.mp4 --depth-mode ai

# 颜色深度计算
python depth_video_converter.py input.jpg output.jpg --depth-mode color_based

# 自定义颜色映射
python depth_video_converter.py input.jpg output.jpg --depth-mode custom --custom-color-map color_map.json
```

#### Web界面方式
```bash
# 启动Web界面
python gradio_app.py
```
然后在浏览器中访问显示的URL，享受可视化操作体验。

## 🎯 深度计算模式

### 1. AI模式 (`ai`)
使用深度学习模型进行深度估计（传统方式）

```bash
python depth_video_converter.py input.mp4 output.mp4 --depth-mode ai --model Intel/dpt-large
```

**支持的模型：**
- `Intel/dpt-large`: 高精度（推荐）
- `Intel/dpt-hybrid-midas`: 平衡精度和速度
- `facebook/dpt-dinov2-small-kitti`: 快速处理

### 2. 颜色深度模式 (`color_based`)
基于图像颜色分析计算深度，主色调区域获得更高深度值

```bash
python depth_video_converter.py input.jpg output.jpg --depth-mode color_based --dominant-weight 0.8
```

**参数说明：**
- `--dominant-weight`: 主色调权重 (0.1-1.0)
- `--depth-range`: 深度范围 (min,max)
- `--similarity-threshold`: 颜色相似度阈值

### 3. 自定义映射模式 (`custom`)
使用预定义的颜色-深度映射

```bash
python depth_video_converter.py input.jpg output.jpg --depth-mode custom --custom-color-map color_map.json
```

## 📁 配置文件

### color_map.json - 颜色深度映射
```json
{
    "红色": 200,
    "绿色": 150,
    "蓝色": 100,
    "说明": "数字越大深度越远，黑色区域保持为0"
}
```

### color_config.json - 颜色深度配置
```json
{
    "主色调权重": 0.7,
    "深度范围": [50, 200],
    "说明": "数字越大深度越远，黑色区域保持为0"
}
```

## 🎬 输出格式

### 1. side_by_side（并排显示）
原始图像和深度图并排显示，便于对比

### 2. depth_only（深度图单独显示）
只显示深度图，使用彩色映射显示深度信息

### 3. overlay（叠加显示）
将深度图叠加到原始图像上，透明度可调

## 💻 编程接口

### 基本使用
```python
from depth_video_converter import DepthVideoConverter

# AI深度估计
converter = DepthVideoConverter(
    model_name="Intel/dpt-large",
    device="auto",
    output_format="side_by_side",
    depth_mode="ai"
)

# 处理视频
success = converter.process_video("input.mp4", "output.mp4")

# 处理图像
success = converter.process_image("input.jpg", "output.jpg")
```

### 颜色深度计算
```python
# 颜色深度模式
converter = DepthVideoConverter(
    depth_mode="color_based",
    color_depth_config={
        "dominant_weight": 0.8,
        "depth_range": [50, 200],
        "similarity_threshold": 30
    }
)

# 自定义颜色映射
converter = DepthVideoConverter(
    depth_mode="custom",
    custom_color_map={
        "红色": 200,
        "绿色": 150,
        "蓝色": 100
    }
)
```

## 🌐 Web界面功能

### 启动Web界面
```bash
python gradio_app.py
```

### 界面特点
- 🎨 **三种深度模式切换** - AI/颜色/自定义模式
- 🎛️ **实时参数调整** - 滑块和输入框实时调整
- 🖼️ **在线预览** - 图像和视频处理结果即时显示
- 📱 **响应式设计** - 支持各种屏幕尺寸
- 🔄 **智能界面** - 根据模式自动显示相关参数

### 使用流程
1. 选择深度计算模式
2. 调整相关参数（AI模式选择模型，颜色模式调整权重等）
3. 上传图像或视频
4. 点击处理按钮
5. 查看和下载结果

## ⚙️ 命令行参数

### 基本参数
```bash
python depth_video_converter.py [输入文件] [输出文件] [选项]

选项:
  --depth-mode MODE        深度计算模式 (ai/color_based/custom)
  --model MODEL           深度估计模型名称
  --device DEVICE         计算设备 (auto/cuda/cpu)
  --format FORMAT         输出格式 (side_by_side/depth_only/overlay)
  --black-threshold N     黑色背景检测阈值 (默认: 30)
```

### 颜色深度参数
```bash
  --dominant-weight F     主色调权重 (0.1-1.0, 默认: 0.7)
  --depth-range MIN,MAX   深度范围 (默认: 50,200)
  --similarity-threshold N 颜色相似度阈值 (默认: 30)
  --color-config FILE     颜色深度配置文件
  --custom-color-map FILE 自定义颜色映射文件
```

### 视频处理参数
```bash
  --start-frame N         起始帧 (默认: 0)
  --max-frames N          最大处理帧数 (默认: 全部)
  --fps FPS              输出视频帧率 (默认: 保持原帧率)
  --force-images         强制输出为图像序列
  --auto-convert         自动使用FFmpeg转换
```

## 📊 性能优化

### 1. GPU加速
- **CUDA加速**: 使用`--device cuda`大幅提升处理速度（5-10倍）
- **自动检测**: 使用`--device auto`自动选择最佳设备
- **内存优化**: GPU模式需要4-8GB VRAM

### 2. 硬件优化
- **GPU加速**: 使用`--device cuda`大幅提升AI模式处理速度
- **内存管理**: 长视频建议使用`--max-frames`分段处理
- **存储优化**: 使用SSD提升I/O性能

### 3. 模型选择
- **高精度**: `Intel/dpt-large` - 适合高质量需求
- **平衡**: `Intel/dpt-hybrid-midas` - 精度和速度平衡
- **高速度**: `facebook/dpt-dinov2-small-kitti` - 快速处理

### 4. 颜色深度优化
- **主色调权重**: 0.7-0.9适合大多数场景
- **深度范围**: [50,200]提供良好的对比度
- **相似度阈值**: 30-50适合大多数图像

### 5. 高级加速（可选）
- **TensorRT**: 需要额外配置，适合生产环境
- **ONNX Runtime**: 需要CUDA支持
- **模型量化**: 减少内存占用

## 🔧 高级功能

### 批量处理
```python
import os
from depth_video_converter import DepthVideoConverter

converter = DepthVideoConverter(depth_mode="color_based")

# 批量处理视频
video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
for i, video_file in enumerate(video_files):
    if os.path.exists(video_file):
        output_file = f"depth_video_{i+1}.mp4"
        converter.process_video(video_file, output_file)
```

### 视频片段处理
```python
# 处理视频的特定片段
converter.process_video(
    input_path="long_video.mp4",
    output_path="segment.mp4",
    start_frame=100,
    max_frames=200,
    fps=24
)
```

### 自定义颜色映射
```python
# 创建自定义颜色映射
custom_map = {
    "红色": 220,      # 红色最远
    "绿色": 150,      # 绿色中等
    "蓝色": 80,       # 蓝色最近
    "黄色": 180,      # 黄色较远
    "紫色": 60,       # 紫色很近
}

converter = DepthVideoConverter(
    depth_mode="custom",
    custom_color_map=custom_map
)
```

## 🎨 颜色深度算法原理

### 1. 颜色分析
- 使用K-means聚类提取图像中的主要颜色
- 计算每种颜色在非黑色区域的占比
- 识别主色调（占比最高的颜色）

### 2. 深度映射
- **主色调区域**: 获得较高深度值（基于权重）
- **其他颜色区域**: 根据占比和相似度计算深度
- **黑色区域**: 深度始终为0（保持不变）

### 3. 自定义映射
- 计算像素与预定义颜色的相似度
- 将最相似颜色的深度值分配给像素
- 支持中英文颜色名称

## 🐛 常见问题

### Q: 处理速度很慢怎么办？
A: 
- 使用GPU加速：`--device cuda`
- 选择更快的模型：`--model facebook/dpt-dinov2-small-kitti`
- 使用颜色深度模式：`--depth-mode color_based`
- 减少处理帧数：`--max-frames 100`

### Q: 内存不足怎么办？
A: 
- 使用CPU：`--device cpu`
- 减少处理帧数
- 选择更小的模型
- 使用颜色深度模式（内存占用更少）

### Q: 颜色深度效果不理想怎么办？
A: 
- 调整主色调权重：`--dominant-weight 0.8`
- 修改深度范围：`--depth-range 30,220`
- 使用自定义颜色映射
- 调整黑色阈值：`--black-threshold 20`

### Q: Web界面无法访问怎么办？
A: 
- 检查端口是否被占用
- 尝试使用`--server-name 127.0.0.1`
- 检查防火墙设置
- 确保安装了gradio：`pip install gradio`

## 📁 项目结构

```
depth/
├── depth_video_converter.py    # 主程序（支持AI+颜色深度）
├── gradio_app.py              # Web界面（支持所有功能）
├── images_to_video.py         # 图像序列转视频工具
├── example.py                 # 使用示例
├── install.py                 # 快速安装脚本
├── requirements.txt           # 依赖包列表
├── README.md                  # 完整说明文档
├── GPU_ACCELERATION.md        # GPU加速详细说明
├── color_map.json            # 颜色深度映射配置
└── color_config.json         # 颜色深度参数配置
```

## 🔄 更新日志

### v2.0.0 (最新)
- ✨ 新增颜色深度计算功能
- ✨ 新增Web界面支持
- ✨ 支持自定义颜色映射
- ✨ 保持黑色区域不变
- ✨ 主色调深度增强
- 🔧 优化界面和用户体验

### v1.0.0
- 🎉 初始版本
- ✨ 支持AI深度估计
- ✨ 支持多种输出格式
- ✨ GPU加速支持

## 📄 许可证

本项目基于MIT许可证开源。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 支持

如果您在使用过程中遇到问题，请：
1. 查看本文档的常见问题部分
2. 检查依赖是否正确安装
3. 提交Issue描述问题

---

**享受深度视频转换的乐趣！** 🎉