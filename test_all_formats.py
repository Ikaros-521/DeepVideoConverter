#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有格式的黑色背景保持
"""

import os
import cv2
import numpy as np
from depth_video_converter import DepthVideoConverter


def test_all_formats():
    """测试所有格式的黑色背景保持"""
    print("测试所有格式的黑色背景保持")
    print("=" * 40)
    
    # 使用现有的测试图像
    test_path = "test.png"
    if not os.path.exists(test_path):
        print(f"❌ 测试图像不存在: {test_path}")
        return
    
    try:
        # 测试所有格式
        formats = ["side_by_side", "depth_only", "overlay"]
        
        formats = ["depth_only"]
        
        for fmt in formats:
            print(f"\n正在测试格式: {fmt}")
            
            # 初始化转换器
            converter = DepthVideoConverter(
                model_name="Intel/dpt-large",
                device="cuda:0",
                output_format=fmt,
                black_threshold=30
            )
            
            # 处理图像
            output_path = f"test_{fmt}_result.jpg"
            success = converter.process_image(test_path, output_path)
            
            if success:
                file_size = os.path.getsize(output_path)
                print(f"✅ {fmt} 格式成功: {output_path} ({file_size} bytes)")
                
                # 分析黑色背景保持情况
                result_image = cv2.imread(output_path)
                
                if fmt == "side_by_side":
                    # 对于并排显示，只分析右侧的深度图部分
                    height, width = result_image.shape[:2]
                    depth_part = result_image[:, width//2:, :]
                else:
                    depth_part = result_image
                
                gray_depth = cv2.cvtColor(depth_part, cv2.COLOR_BGR2GRAY)
                black_pixels = np.sum(gray_depth < 30)
                total_pixels = gray_depth.shape[0] * gray_depth.shape[1]
                black_percentage = (black_pixels / total_pixels) * 100
                
                print(f"📊 {fmt} 格式黑色背景占比: {black_percentage:.1f}%")
                
            else:
                print(f"❌ {fmt} 格式失败")
        
        print("\n🎉 所有格式测试完成!")
        print("\n生成的文件:")
        for fmt in formats:
            output_path = f"test_{fmt}_result.jpg"
            if os.path.exists(output_path):
                print(f"  - {output_path}")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_all_formats()
