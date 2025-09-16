#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度视频转换器 - 快速安装脚本
"""

import subprocess
import sys
import os

def install_requirements():
    """安装requirements.txt中的依赖"""
    print("📦 正在安装依赖包...")
    
    try:
        # 检查requirements.txt是否存在
        if not os.path.exists("requirements.txt"):
            print("❌ requirements.txt 文件不存在")
            return False
        
        # 安装依赖
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def check_installation():
    """检查安装是否成功"""
    print("\n🔍 检查安装状态...")
    
    required_packages = [
        ("torch", "torch"),
        ("opencv-python", "cv2"),
        ("transformers", "transformers"), 
        ("scikit-learn", "sklearn"),
        ("gradio", "gradio"),
        ("numpy", "numpy"),
        ("pillow", "PIL")
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️ 缺少依赖: {', '.join(missing_packages)}")
        return False
    else:
        print("\n🎉 所有依赖安装成功！")
        return True

def main():
    """主函数"""
    print("🚀 深度视频转换器 - 快速安装")
    print("=" * 50)
    
    # 安装依赖
    if install_requirements():
        # 检查安装
        if check_installation():
            print("\n📝 安装完成！现在可以使用:")
            print("1. python gradio_app.py - 启动Web界面")
            print("2. python depth_video_converter.py - 命令行处理")
            print("3. python example.py - 查看使用示例")
            print("\n🎉 享受深度视频转换的乐趣！")
        else:
            print("\n❌ 安装检查失败，请手动安装缺少的依赖")
    else:
        print("\n❌ 安装失败，请检查网络连接和Python环境")

if __name__ == "__main__":
    main()
