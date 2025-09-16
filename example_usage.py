#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度视频转换程序 - 使用示例
"""

import os
import sys
import torch
from depth_video_converter import DepthVideoConverter


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 初始化转换器
    converter = DepthVideoConverter(
        model_name="Intel/dpt-large",  # 使用Intel的DPT大模型
        device="auto",                 # 自动选择设备
        output_format="side_by_side"   # 并排显示格式
    )
    
    # 处理视频
    input_video = "input_video.mp4"
    output_video = "output_depth_video.mp4"
    
    if os.path.exists(input_video):
        success = converter.process_video(input_video, output_video)
        if success:
            print(f"视频转换成功: {output_video}")
        else:
            print("视频转换失败")
    else:
        print(f"输入视频不存在: {input_video}")


def example_image_processing():
    """图像处理示例"""
    print("\n=== 图像处理示例 ===")
    
    converter = DepthVideoConverter(
        model_name="Intel/dpt-large",
        output_format="overlay"  # 叠加显示
    )
    
    input_image = "input_image.jpg"
    output_image = "output_depth_image.jpg"
    
    if os.path.exists(input_image):
        success = converter.process_image(input_image, output_image)
        if success:
            print(f"图像转换成功: {output_image}")
        else:
            print("图像转换失败")
    else:
        print(f"输入图像不存在: {input_image}")


def example_custom_settings():
    """自定义设置示例"""
    print("\n=== 自定义设置示例 ===")
    
    # 使用不同的深度估计模型
    models = [
        "Intel/dpt-large",      # Intel DPT大模型
        "Intel/dpt-hybrid-midas", # Intel DPT混合模型
        "facebook/dpt-dinov2-small-kitti", # Facebook DPT小模型
    ]
    
    for model_name in models:
        print(f"尝试使用模型: {model_name}")
        try:
            converter = DepthVideoConverter(
                model_name=model_name,
                device="cpu",  # 强制使用CPU
                output_format="depth_only"  # 只显示深度图
            )
            print(f"模型 {model_name} 加载成功")
        except Exception as e:
            print(f"模型 {model_name} 加载失败: {e}")


def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    converter = DepthVideoConverter()
    
    # 处理多个视频文件
    video_files = [
        "video1.mp4",
        "video2.mp4", 
        "video3.mp4"
    ]
    
    for i, video_file in enumerate(video_files):
        if os.path.exists(video_file):
            output_file = f"depth_video_{i+1}.mp4"
            print(f"处理视频 {i+1}: {video_file}")
            
            success = converter.process_video(
                input_path=video_file,
                output_path=output_file,
                start_frame=0,      # 从第0帧开始
                max_frames=100,     # 最多处理100帧
                fps=24              # 输出24fps
            )
            
            if success:
                print(f"  -> 成功: {output_file}")
            else:
                print(f"  -> 失败: {video_file}")
        else:
            print(f"文件不存在: {video_file}")


def example_advanced_usage():
    """高级使用示例"""
    print("\n=== 高级使用示例 ===")
    
    # 创建自定义转换器
    converter = DepthVideoConverter(
        model_name="Intel/dpt-large",
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_format="side_by_side"
    )
    
    # 处理视频的特定片段
    input_video = "long_video.mp4"
    if os.path.exists(input_video):
        # 从第100帧开始，处理200帧
        success = converter.process_video(
            input_path=input_video,
            output_path="video_segment_depth.mp4",
            start_frame=100,
            max_frames=200,
            fps=30
        )
        
        if success:
            print("视频片段深度转换完成")
        else:
            print("视频片段深度转换失败")


if __name__ == "__main__":
    print("深度视频转换程序 - 使用示例")
    print("=" * 50)
    
    # 运行各种示例
    example_basic_usage()
    example_image_processing()
    example_custom_settings()
    example_batch_processing()
    example_advanced_usage()
    
    print("\n示例运行完成!")
    print("\n使用方法:")
    print("python depth_video_converter.py input.mp4 output.mp4")
    print("python depth_video_converter.py input.jpg output.jpg --format overlay")
    print("python depth_video_converter.py input.mp4 output.mp4 --max-frames 100 --fps 24")
