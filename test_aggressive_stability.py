#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试激进的时序稳定性功能
"""

import cv2
import numpy as np
from depth_video_converter import DepthVideoConverter
import os

def create_complex_test_video():
    """创建复杂的测试视频"""
    test_video_path = "test_aggressive_stability_input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))
    
    # 创建复杂的移动场景
    for i in range(120):  # 4秒视频
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 移动的红色矩形（快速移动）
        x1 = int(50 + i * 4)
        y1 = 100
        cv2.rectangle(frame, (x1, y1), (x1 + 100, y1 + 100), (0, 0, 255), -1)
        
        # 移动的绿色圆形（中等速度）
        x2 = int(200 + i * 2.5)
        y2 = 200
        cv2.circle(frame, (x2, y2), 50, (0, 255, 0), -1)
        
        # 移动的蓝色三角形（慢速移动）
        x3 = int(400 - i * 1)
        y3 = 300
        pts = np.array([[x3, y3], [x3+60, y3], [x3+30, y3-60]], np.int32)
        cv2.fillPoly(frame, [pts], (255, 0, 0))
        
        # 移动的黄色椭圆（复杂运动）
        x4 = int(300 + 50 * np.sin(i * 0.1))
        y4 = int(150 + 30 * np.cos(i * 0.15))
        cv2.ellipse(frame, (x4, y4), (40, 20), 0, 0, 360, (0, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ 复杂测试视频已生成: {test_video_path}")
    return test_video_path

def test_aggressive_stability():
    """测试激进的时序稳定性功能"""
    print("🧪 测试激进的时序稳定性功能...")
    
    # 创建测试视频
    test_video_path = create_complex_test_video()
    
    # 测试1: 激进的时序稳定性
    print("\n🔧 测试1: 激进的时序稳定性")
    converter_aggressive = DepthVideoConverter(
        depth_mode="ai",
        temporal_stability=True,
        global_normalization=True,
        output_format="depth_only"
    )
    
    success1 = converter_aggressive.process_video(
        test_video_path, 
        "test_aggressive_stable_output.mp4",
        max_frames=90
    )
    
    if success1:
        print("✅ 激进时序稳定性测试成功")
    else:
        print("❌ 激进时序稳定性测试失败")
    
    # 测试2: 原始算法（无稳定性）
    print("\n🔧 测试2: 原始算法（无稳定性）")
    converter_original = DepthVideoConverter(
        depth_mode="ai",
        temporal_stability=False,
        global_normalization=False,
        output_format="depth_only"
    )
    
    success2 = converter_original.process_video(
        test_video_path, 
        "test_original_output.mp4",
        max_frames=90
    )
    
    if success2:
        print("✅ 原始算法测试成功")
    else:
        print("❌ 原始算法测试失败")
    
    # 清理测试文件
    if os.path.exists(test_video_path):
        os.remove(test_video_path)
    
    print("\n📊 测试完成！")
    print("请比较以下文件：")
    print("- test_aggressive_stable_output.mp4 (激进稳定性)")
    print("- test_original_output.mp4 (原始算法)")
    print("\n激进稳定性版本应该有：")
    print("✅ 几乎无颜色突变")
    print("✅ 深度值高度一致")
    print("✅ 物体运动时颜色保持稳定")
    print("✅ 整体深度分布稳定")

if __name__ == "__main__":
    test_aggressive_stability()
