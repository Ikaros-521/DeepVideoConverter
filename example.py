#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
颜色深度转换示例
"""

import os
from depth_video_converter import DepthVideoConverter

def 自动颜色深度():
    """自动颜色深度计算"""
    print("=== 自动颜色深度计算 ===")
    
    converter = DepthVideoConverter(depth_mode="color_based")
    
    if os.path.exists("test.png"):
        success = converter.process_image("test.png", "输出_自动深度.jpg")
        if success:
            print("✅ 自动颜色深度处理完成!")
        else:
            print("❌ 处理失败")
    else:
        print("❌ 找不到 test.png 文件")

def 自定义颜色深度():
    """自定义颜色深度"""
    print("\n=== 自定义颜色深度 ===")
    
    # 简单的颜色映射
    颜色深度 = {
        "红色": 200,    # 红色最远
        "绿色": 150,    # 绿色中等
        "蓝色": 100,    # 蓝色最近
    }
    
    converter = DepthVideoConverter(
        depth_mode="custom",
        custom_color_map=颜色深度
    )
    
    if os.path.exists("test.png"):
        success = converter.process_image("test.png", "输出_自定义深度.jpg")
        if success:
            print("✅ 自定义颜色深度处理完成!")
        else:
            print("❌ 处理失败")
    else:
        print("❌ 找不到 test.png 文件")

def main():
    """主函数"""
    print("🎨 颜色深度转换示例")
    print("=" * 30)
    
    自动颜色深度()
    自定义颜色深度()
    
    print("\n📝 使用方法:")
    print("1. 自动颜色深度:")
    print("   python depth_video_converter.py 输入.jpg 输出.jpg --depth-mode color_based")
    print()
    print("2. 自定义颜色:")
    print("   python depth_video_converter.py 输入.jpg 输出.jpg --depth-mode custom --custom-color-map color_map.json")
    print()
    print("3. 调整主色调权重:")
    print("   python depth_video_converter.py 输入.jpg 输出.jpg --depth-mode color_based --dominant-weight 0.8")

if __name__ == "__main__":
    main()
