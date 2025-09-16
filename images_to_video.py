#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像序列转视频工具
使用FFmpeg将图像序列重新合成为视频

作者: AI Assistant
日期: 2024
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import glob

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageSequenceToVideo:
    """图像序列转视频转换器"""
    
    def __init__(self):
        """初始化转换器"""
        self.ffmpeg_path = self._find_ffmpeg()
        if not self.ffmpeg_path:
            logger.warning("未找到FFmpeg，请确保已安装FFmpeg并添加到PATH环境变量")
    
    def _find_ffmpeg(self):
        """跨平台查找FFmpeg可执行文件"""
        import platform
        import shutil
        
        # 首先尝试在PATH中查找
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        
        # 如果PATH中找不到，尝试常见路径
        if platform.system() == 'Windows':
            possible_paths = [
                'ffmpeg.exe',
                r'C:\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
                r'C:\tools\ffmpeg\bin\ffmpeg.exe',
            ]
        elif platform.system() == 'Darwin':  # macOS
            possible_paths = [
                '/usr/local/bin/ffmpeg',
                '/opt/homebrew/bin/ffmpeg',
                '/usr/bin/ffmpeg',
            ]
        else:  # Linux
            possible_paths = [
                '/usr/bin/ffmpeg',
                '/usr/local/bin/ffmpeg',
                '/opt/ffmpeg/bin/ffmpeg',
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def convert_images_to_video(self, 
                               input_dir: str, 
                               output_path: str,
                               fps: float = 30.0,
                               codec: str = 'libx264',
                               quality: str = 'medium',
                               start_number: int = 0,
                               pattern: str = 'frame_%06d.jpg') -> bool:
        """
        将图像序列转换为视频
        
        Args:
            input_dir: 图像序列目录
            output_path: 输出视频路径
            fps: 输出视频帧率
            codec: 视频编码器
            quality: 视频质量 (low, medium, high, lossless)
            start_number: 起始帧号
            pattern: 文件名模式
            
        Returns:
            bool: 转换是否成功
        """
        if not self.ffmpeg_path:
            logger.error("FFmpeg未找到，无法转换视频")
            return False
        
        if not os.path.exists(input_dir):
            logger.error(f"输入目录不存在: {input_dir}")
            return False
        
        # 检查图像文件
        image_files = self._get_image_files(input_dir, pattern)
        if not image_files:
            logger.error(f"在目录 {input_dir} 中未找到匹配的图像文件")
            return False
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 构建FFmpeg命令
        cmd = self._build_ffmpeg_command(
            input_dir, output_path, fps, codec, quality, start_number, pattern
        )
        
        logger.info(f"执行FFmpeg命令: {' '.join(cmd)}")
        
        try:
            # 跨平台兼容的subprocess调用
            import platform
            if platform.system() == 'Windows':
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, shell=True)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"视频转换成功: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg执行失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg执行超时")
            return False
        except Exception as e:
            logger.error(f"执行FFmpeg时发生错误: {e}")
            return False
    
    def _get_image_files(self, input_dir: str, pattern: str) -> list:
        """获取图像文件列表"""
        import re
        
        # 如果pattern包含%06d这样的占位符，转换为实际的文件名模式
        if '%06d' in pattern:
            # 将frame_%06d.jpg转换为frame_*.jpg
            actual_pattern = pattern.replace('%06d', '*')
        elif '%d' in pattern:
            # 将frame_%d.jpg转换为frame_*.jpg
            actual_pattern = pattern.replace('%d', '*')
        else:
            # 直接使用pattern
            actual_pattern = pattern
        
        # 搜索匹配的文件
        search_pattern = os.path.join(input_dir, actual_pattern)
        image_files = glob.glob(search_pattern)
        
        # 按文件名排序（数字顺序）
        def natural_sort_key(filename):
            # 提取文件名中的数字进行排序
            numbers = re.findall(r'\d+', os.path.basename(filename))
            return [int(num) for num in numbers] if numbers else [0]
        
        return sorted(image_files, key=natural_sort_key)
    
    def _build_ffmpeg_command(self, input_dir, output_path, fps, codec, quality, start_number, pattern):
        """构建FFmpeg命令"""
        cmd = [self.ffmpeg_path]
        
        # 搜索图像文件并创建文件列表
        image_files = self._get_image_files(input_dir, pattern)
        if not image_files:
            logger.error(f"未找到图像文件")
            return []
        
        # 如果用户指定了fps为0，保持原设置（图像序列通常不需要fps检查）
        # 只有当fps为None时才使用默认值
        if fps is None:
            fps = 30.0
        
        # 使用concat demuxer方式
        concat_file = os.path.join(input_dir, "filelist.txt")
        with open(concat_file, 'w', encoding='utf-8') as f:
            for img_file in image_files:
                img_path = img_file.replace('\\', '/')
                f.write(f"file '{img_path}'\n")
                f.write(f"duration {1/fps}\n")
            # 最后一行不需要duration
            last_file_path = image_files[-1].replace('\\', '/')
            f.write(f"file '{last_file_path}'\n")
        
        cmd.extend(['-f', 'concat'])
        cmd.extend(['-safe', '0'])
        cmd.extend(['-i', concat_file.replace('\\', '/')])
        
        # 视频编码参数
        if codec == 'libx264':
            cmd.extend(['-c:v', 'libx264'])
            
            # 质量设置
            if quality == 'low':
                cmd.extend(['-crf', '28', '-preset', 'fast'])
            elif quality == 'medium':
                cmd.extend(['-crf', '23', '-preset', 'medium'])
            elif quality == 'high':
                cmd.extend(['-crf', '18', '-preset', 'slow'])
            elif quality == 'lossless':
                cmd.extend(['-crf', '0', '-preset', 'veryslow'])
            else:
                cmd.extend(['-crf', '23', '-preset', 'medium'])
                
        elif codec == 'libx265':
            cmd.extend(['-c:v', 'libx265'])
            if quality == 'low':
                cmd.extend(['-crf', '30', '-preset', 'fast'])
            elif quality == 'medium':
                cmd.extend(['-crf', '25', '-preset', 'medium'])
            elif quality == 'high':
                cmd.extend(['-crf', '20', '-preset', 'slow'])
            elif quality == 'lossless':
                cmd.extend(['-crf', '0', '-preset', 'veryslow'])
            else:
                cmd.extend(['-crf', '25', '-preset', 'medium'])
        
        elif codec == 'libvpx-vp9':
            cmd.extend(['-c:v', 'libvpx-vp9'])
            if quality == 'low':
                cmd.extend(['-crf', '35', '-b:v', '0'])
            elif quality == 'medium':
                cmd.extend(['-crf', '30', '-b:v', '0'])
            elif quality == 'high':
                cmd.extend(['-crf', '25', '-b:v', '0'])
            elif quality == 'lossless':
                cmd.extend(['-crf', '0', '-b:v', '0'])
            else:
                cmd.extend(['-crf', '30', '-b:v', '0'])
        
        # 其他参数
        cmd.extend(['-pix_fmt', 'yuv420p'])  # 确保兼容性
        cmd.extend(['-movflags', '+faststart'])  # 优化网络播放
        
        # 输出文件
        cmd.append(output_path)
        
        return cmd
    
    def get_supported_codecs(self):
        """获取支持的编码器列表"""
        if not self.ffmpeg_path:
            return []
        
        try:
            import platform
            if platform.system() == 'Windows':
                result = subprocess.run([self.ffmpeg_path, '-encoders'], 
                                      capture_output=True, text=True, timeout=10, shell=True)
            else:
                result = subprocess.run([self.ffmpeg_path, '-encoders'], 
                                      capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                codecs = []
                for line in lines:
                    if 'libx264' in line or 'libx265' in line or 'libvpx' in line:
                        codecs.append(line.strip())
                return codecs
        except:
            pass
        
        return ['libx264', 'libx265', 'libvpx-vp9']


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图像序列转视频工具")
    parser.add_argument("input_dir", help="图像序列目录")
    parser.add_argument("output", help="输出视频路径")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="输出视频帧率 (默认: 30.0)")
    parser.add_argument("--codec", default="libx264",
                       choices=["libx264", "libx265", "libvpx-vp9"],
                       help="视频编码器 (默认: libx264)")
    parser.add_argument("--quality", default="medium",
                       choices=["low", "medium", "high", "lossless"],
                       help="视频质量 (默认: medium)")
    parser.add_argument("--start-number", type=int, default=0,
                       help="起始帧号 (默认: 0)")
    parser.add_argument("--pattern", default="frame_%06d.jpg",
                       help="文件名模式 (默认: frame_%%06d.jpg)")
    parser.add_argument("--list-codecs", action="store_true",
                       help="列出支持的编码器")
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = ImageSequenceToVideo()
    
    if args.list_codecs:
        codecs = converter.get_supported_codecs()
        print("支持的编码器:")
        for codec in codecs:
            print(f"  - {codec}")
        return
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    # 执行转换
    success = converter.convert_images_to_video(
        input_dir=args.input_dir,
        output_path=args.output,
        fps=args.fps,
        codec=args.codec,
        quality=args.quality,
        start_number=args.start_number,
        pattern=args.pattern
    )
    
    if success:
        logger.info("转换完成!")
    else:
        logger.error("转换失败!")


if __name__ == "__main__":
    main()
