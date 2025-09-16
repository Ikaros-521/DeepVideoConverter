#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度视频转换程序
将普通视频转换为包含深度信息的视频

作者: AI Assistant
日期: 2024
"""

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List
import logging
from tqdm import tqdm
import warnings
import traceback
import subprocess
from collections import Counter
from sklearn.cluster import KMeans
import json
warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DepthVideoConverter:
    """深度视频转换器"""
    
    def __init__(self, 
                 model_name: str = "Intel/dpt-large",
                 device: str = "auto",
                 output_format: str = "side_by_side",
                 black_threshold: int = 30,
                 depth_mode: str = "ai",
                 color_depth_config: Optional[Dict] = None,
                 custom_color_map: Optional[Dict] = None,
                 temporal_stability: bool = True,
                 global_normalization: bool = True):
        """
        初始化深度视频转换器
        
        Args:
            model_name: 深度估计模型名称
            device: 计算设备 ("auto", "cuda", "cpu")
            output_format: 输出格式 ("side_by_side", "depth_only", "overlay")
            black_threshold: 黑色背景检测阈值
            depth_mode: 深度计算模式 ("ai", "color_based", "custom")
            color_depth_config: 颜色深度配置
            custom_color_map: 自定义颜色深度映射
            temporal_stability: 是否启用时序稳定性
            global_normalization: 是否使用全局深度归一化
        """
        self.model_name = model_name
        self.output_format = output_format
        self.black_threshold = black_threshold
        self.depth_mode = depth_mode
        self.color_depth_config = color_depth_config or {}
        self.custom_color_map = custom_color_map or {}
        self.temporal_stability = temporal_stability
        self.global_normalization = global_normalization
        
        # 时序稳定性相关变量
        self.previous_depth_map = None
        self.global_depth_min = None
        self.global_depth_max = None
        self.depth_history = []  # 存储最近几帧的深度图用于平滑
        self.previous_image = None  # 存储前一帧的图像用于颜色一致性
        
        # 深度值锁定机制
        self.depth_locks = {}  # 存储已锁定的深度值
        self.color_depth_mapping = {}  # 颜色到深度的映射
        self.lock_threshold = 0.85  # 锁定阈值
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"深度计算模式: {self.depth_mode}")
        
        # 初始化深度估计模型（仅在AI模式下需要）
        if self.depth_mode == "ai":
            self._load_depth_model()
        
        # 检查FFmpeg可用性
        self.ffmpeg_available = self._check_ffmpeg()
    
    def _load_depth_model(self):
        """加载深度估计模型"""
        try:
            from transformers import pipeline
            logger.info(f"正在加载深度估计模型: {self.model_name}")
            
            self.depth_estimator = pipeline(
                "depth-estimation", 
                model=self.model_name,
                device=0 if self.device.type == "cuda" else -1
            )
            logger.info("深度估计模型加载成功")
            
        except ImportError:
            logger.error("需要安装 transformers 库: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"加载深度估计模型失败: {e}")
            raise
    
    def _check_ffmpeg(self):
        """检查FFmpeg是否可用"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("FFmpeg可用")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        logger.warning("FFmpeg不可用，无法将图像序列转换为视频")
        return False
    
    def _analyze_colors(self, image: np.ndarray, n_colors: int = 8) -> Dict:
        """
        分析图像中的主要颜色及其占比
        
        Args:
            image: 输入图像 (H, W, 3)
            n_colors: 要提取的主要颜色数量
            
        Returns:
            color_info: 包含主要颜色和占比的字典
        """
        try:
            # 检测黑色背景区域
            black_mask = (image[:, :, 0] < self.black_threshold) & \
                        (image[:, :, 1] < self.black_threshold) & \
                        (image[:, :, 2] < self.black_threshold)
            
            # 只分析非黑色区域
            non_black_pixels = image[~black_mask]
            
            if len(non_black_pixels) == 0:
                logger.warning("图像中没有非黑色像素")
                return {"colors": [], "ratios": [], "dominant_color": None}
            
            # 使用K-means聚类提取主要颜色
            kmeans = KMeans(n_clusters=min(n_colors, len(non_black_pixels)), 
                          random_state=42, n_init=10)
            kmeans.fit(non_black_pixels)
            
            # 获取聚类中心和标签
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # 计算每种颜色的占比
            label_counts = Counter(labels)
            total_pixels = len(non_black_pixels)
            ratios = [count / total_pixels for count in label_counts.values()]
            
            # 按占比排序
            sorted_indices = np.argsort(ratios)[::-1]
            sorted_colors = colors[sorted_indices]
            sorted_ratios = [ratios[i] for i in sorted_indices]
            
            # 获取主色调
            dominant_color = sorted_colors[0] if len(sorted_colors) > 0 else None
            
            color_info = {
                "colors": sorted_colors.tolist(),
                "ratios": sorted_ratios,
                "dominant_color": dominant_color.tolist() if dominant_color is not None else None,
                "total_colors": len(sorted_colors)
            }
            
            logger.info(f"检测到 {len(sorted_colors)} 种主要颜色")
            logger.info(f"主色调: RGB{dominant_color} (占比: {sorted_ratios[0]:.2%})")
            
            return color_info
            
        except Exception as e:
            logger.error(f"颜色分析失败: {e}")
            return {"colors": [], "ratios": [], "dominant_color": None}
    
    def _calculate_color_depth(self, image: np.ndarray, color_info: Dict) -> np.ndarray:
        """
        基于颜色信息计算深度图
        
        Args:
            image: 输入图像 (H, W, 3)
            color_info: 颜色分析结果
            
        Returns:
            depth_map: 基于颜色的深度图 (H, W)
        """
        try:
            height, width = image.shape[:2]
            depth_map = np.zeros((height, width), dtype=np.float32)
            
            # 检测黑色背景区域
            black_mask = (image[:, :, 0] < self.black_threshold) & \
                        (image[:, :, 1] < self.black_threshold) & \
                        (image[:, :, 2] < self.black_threshold)
            
            # 黑色区域深度设为0
            depth_map[black_mask] = 0
            
            if not color_info["colors"]:
                logger.warning("没有检测到颜色信息，使用默认深度")
                depth_map[~black_mask] = 128  # 默认中等深度
                return depth_map
            
            # 获取颜色配置
            config = self.color_depth_config
            dominant_weight = config.get("dominant_weight", 0.7)  # 主色调权重
            color_depth_range = config.get("depth_range", [50, 200])  # 深度范围
            similarity_threshold = config.get("similarity_threshold", 30)  # 颜色相似度阈值
            
            colors = np.array(color_info["colors"])
            ratios = color_info["ratios"]
            dominant_color = np.array(color_info["dominant_color"])
            
            # 为非黑色区域计算深度
            non_black_mask = ~black_mask
            non_black_pixels = image[non_black_mask]
            
            for i, pixel in enumerate(non_black_pixels):
                # 计算与各种颜色的相似度
                distances = np.sqrt(np.sum((colors - pixel) ** 2, axis=1))
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                
                # 如果与主色调相似，给予更高深度
                dominant_distance = np.sqrt(np.sum((dominant_color - pixel) ** 2))
                
                if dominant_distance < similarity_threshold:
                    # 主色调区域，使用较高深度值
                    depth_value = color_depth_range[1] * dominant_weight + \
                                color_depth_range[0] * (1 - dominant_weight)
                else:
                    # 其他颜色区域，根据占比和相似度计算深度
                    color_ratio = ratios[min_distance_idx]
                    similarity_factor = max(0, 1 - min_distance / 255)  # 归一化相似度
                    
                    # 深度值 = 基础深度 + 占比影响 + 相似度影响
                    base_depth = color_depth_range[0]
                    ratio_influence = (color_depth_range[1] - color_depth_range[0]) * color_ratio
                    similarity_influence = (color_depth_range[1] - color_depth_range[0]) * similarity_factor * 0.3
                    
                    depth_value = base_depth + ratio_influence + similarity_influence
                    depth_value = np.clip(depth_value, color_depth_range[0], color_depth_range[1])
                
                # 获取像素在原始图像中的位置
                pixel_idx = np.where(non_black_mask.ravel())[0][i]
                y, x = divmod(pixel_idx, width)
                depth_map[y, x] = depth_value
            
            # 归一化到0-255
            depth_map = np.clip(depth_map, 0, 255).astype(np.uint8)
            
            logger.info(f"基于颜色计算深度完成，深度范围: {depth_map.min()}-{depth_map.max()}")
            return depth_map
            
        except Exception as e:
            logger.error(f"颜色深度计算失败: {e}")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    def _calculate_custom_color_depth(self, image: np.ndarray) -> np.ndarray:
        """
        基于自定义颜色映射计算深度图
        
        Args:
            image: 输入图像 (H, W, 3)
            
        Returns:
            depth_map: 基于自定义映射的深度图 (H, W)
        """
        try:
            height, width = image.shape[:2]
            depth_map = np.zeros((height, width), dtype=np.float32)
            
            # 检测黑色背景区域
            black_mask = (image[:, :, 0] < self.black_threshold) & \
                        (image[:, :, 1] < self.black_threshold) & \
                        (image[:, :, 2] < self.black_threshold)
            
            # 黑色区域深度设为0
            depth_map[black_mask] = 0
            
            if not self.custom_color_map:
                logger.warning("没有自定义颜色映射，使用默认深度")
                depth_map[~black_mask] = 128
                return depth_map
            
            # 为非黑色区域计算深度
            non_black_mask = ~black_mask
            non_black_pixels = image[non_black_mask]
            
            for i, pixel in enumerate(non_black_pixels):
                pixel_rgb = tuple(pixel)
                
                # 查找最匹配的颜色映射
                best_match_depth = 128  # 默认深度
                min_distance = float('inf')
                
                for color_rgb, depth_value in self.custom_color_map.items():
                    if isinstance(color_rgb, str):
                        # 如果是颜色名称，转换为RGB
                        color_rgb = self._color_name_to_rgb(color_rgb)
                    
                    if isinstance(color_rgb, (list, tuple)) and len(color_rgb) == 3:
                        distance = np.sqrt(np.sum((np.array(color_rgb) - pixel) ** 2))
                        if distance < min_distance:
                            min_distance = distance
                            best_match_depth = depth_value
                
                # 获取像素在原始图像中的位置
                pixel_idx = np.where(non_black_mask.ravel())[0][i]
                y, x = divmod(pixel_idx, width)
                depth_map[y, x] = best_match_depth
            
            # 归一化到0-255
            depth_map = np.clip(depth_map, 0, 255).astype(np.uint8)
            
            logger.info(f"基于自定义映射计算深度完成，深度范围: {depth_map.min()}-{depth_map.max()}")
            return depth_map
            
        except Exception as e:
            logger.error(f"自定义颜色深度计算失败: {e}")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    def _color_name_to_rgb(self, color_name: str) -> Tuple[int, int, int]:
        """
        将颜色名称转换为RGB值（支持中英文）
        
        Args:
            color_name: 颜色名称
            
        Returns:
            rgb: RGB元组
        """
        color_map = {
            # 英文颜色名
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'gray': (128, 128, 128),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'brown': (165, 42, 42),
            
            # 中文颜色名
            '红色': (255, 0, 0),
            '绿色': (0, 255, 0),
            '蓝色': (0, 0, 255),
            '黄色': (255, 255, 0),
            '青色': (0, 255, 255),
            '洋红': (255, 0, 255),
            '白色': (255, 255, 255),
            '黑色': (0, 0, 0),
            '灰色': (128, 128, 128),
            '橙色': (255, 165, 0),
            '紫色': (128, 0, 128),
            '粉色': (255, 192, 203),
            '棕色': (165, 42, 42),
        }
        
        return color_map.get(color_name.lower(), (128, 128, 128))
    
    def _apply_temporal_smoothing(self, depth_map: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        应用时序平滑以减少深度值的突变
        
        Args:
            depth_map: 当前帧的深度图
            alpha: 平滑系数 (0-1)，越小越平滑
            
        Returns:
            smoothed_depth_map: 平滑后的深度图
        """
        if not self.temporal_stability or self.previous_depth_map is None:
            self.previous_depth_map = depth_map.copy()
            return depth_map
        
        # 计算帧间差异
        diff = np.abs(depth_map.astype(np.float32) - self.previous_depth_map.astype(np.float32))
        
        # 自适应平滑：差异大的区域使用更强的平滑
        adaptive_alpha = np.where(diff > 30, alpha * 0.5, alpha)  # 差异大时平滑更强
        
        # 应用自适应平滑
        smoothed = adaptive_alpha * self.previous_depth_map.astype(np.float32) + \
                  (1 - adaptive_alpha) * depth_map.astype(np.float32)
        
        # 更新历史记录
        self.previous_depth_map = smoothed.astype(np.uint8).copy()
        
        return smoothed.astype(np.uint8)
    
    def _apply_color_consistent_depth(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        基于颜色一致性应用深度平滑
        
        Args:
            image: 原始图像
            depth_map: 当前深度图
            
        Returns:
            consistent_depth_map: 颜色一致的深度图
        """
        if not self.temporal_stability or self.previous_depth_map is None:
            return depth_map
        
        # 检测黑色背景
        black_mask = (image[:, :, 0] < self.black_threshold) & \
                    (image[:, :, 1] < self.black_threshold) & \
                    (image[:, :, 2] < self.black_threshold)
        
        # 只对非黑色区域应用颜色一致性
        non_black_mask = ~black_mask
        
        if not np.any(non_black_mask):
            return depth_map
        
        # 计算颜色相似度
        current_colors = image[non_black_mask]
        if hasattr(self, 'previous_image') and self.previous_image is not None:
            prev_colors = self.previous_image[non_black_mask]
            
            # 计算颜色差异
            color_diff = np.sqrt(np.sum((current_colors.astype(np.float32) - 
                                       prev_colors.astype(np.float32)) ** 2, axis=1))
            
            # 颜色相似的区域使用更强的时序平滑
            color_similarity = np.exp(-color_diff / 50.0)  # 颜色相似度
            
            # 创建平滑权重
            smooth_weight = np.zeros_like(depth_map, dtype=np.float32)
            smooth_weight[non_black_mask] = color_similarity * 0.8  # 颜色相似时平滑更强
            
            # 应用颜色一致性平滑
            consistent_depth = smooth_weight * self.previous_depth_map.astype(np.float32) + \
                             (1 - smooth_weight) * depth_map.astype(np.float32)
            
            return consistent_depth.astype(np.uint8)
        
        return depth_map
    
    def _apply_depth_statistics_smoothing(self, depth_map: np.ndarray) -> np.ndarray:
        """
        基于深度统计特性的平滑算法
        
        Args:
            depth_map: 当前深度图
            
        Returns:
            smoothed_depth_map: 统计平滑后的深度图
        """
        if not self.temporal_stability or self.previous_depth_map is None:
            return depth_map
        
        # 计算深度值的统计特性
        current_mean = np.mean(depth_map[depth_map > 0])  # 排除黑色背景
        current_std = np.std(depth_map[depth_map > 0])
        
        if hasattr(self, 'depth_stats_history'):
            if len(self.depth_stats_history) > 0:
                # 使用历史统计信息进行平滑
                prev_mean = np.mean([stats['mean'] for stats in self.depth_stats_history[-3:]])  # 最近3帧的平均值
                prev_std = np.mean([stats['std'] for stats in self.depth_stats_history[-3:]])
                
                # 如果统计特性变化太大，使用更强的平滑
                mean_diff = abs(current_mean - prev_mean)
                std_diff = abs(current_std - prev_std)
                
                if mean_diff > 20 or std_diff > 15:  # 统计特性变化较大
                    # 使用历史统计信息调整深度图
                    depth_map_adjusted = depth_map.astype(np.float32)
                    depth_map_adjusted[depth_map > 0] = (depth_map_adjusted[depth_map > 0] - current_mean) * (prev_std / current_std) + prev_mean
                    depth_map_adjusted = np.clip(depth_map_adjusted, 0, 255)
                    
                    # 与前一帧进行混合
                    smoothed = 0.6 * self.previous_depth_map.astype(np.float32) + 0.4 * depth_map_adjusted
                    depth_map = smoothed.astype(np.uint8)
        
        # 保存当前统计信息
        if not hasattr(self, 'depth_stats_history'):
            self.depth_stats_history = []
        
        self.depth_stats_history.append({
            'mean': current_mean,
            'std': current_std
        })
        
        # 只保留最近10帧的统计信息
        if len(self.depth_stats_history) > 10:
            self.depth_stats_history = self.depth_stats_history[-10:]
        
        return depth_map
    
    def _apply_depth_locking_stability(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        基于深度值锁定的激进稳定性算法
        
        Args:
            image: 原始图像
            depth_map: 当前深度图
            
        Returns:
            locked_depth_map: 锁定后的深度图
        """
        if not self.temporal_stability:
            return depth_map
        
        # 检测黑色背景
        black_mask = (image[:, :, 0] < self.black_threshold) & \
                    (image[:, :, 1] < self.black_threshold) & \
                    (image[:, :, 2] < self.black_threshold)
        
        non_black_mask = ~black_mask
        
        if not np.any(non_black_mask):
            return depth_map
        
        # 获取非黑色区域的像素
        non_black_pixels = image[non_black_mask]
        non_black_depths = depth_map[non_black_mask]
        
        # 创建锁定后的深度图
        locked_depth_map = depth_map.copy()
        
        for i, (pixel, current_depth) in enumerate(zip(non_black_pixels, non_black_depths)):
            pixel_tuple = tuple(pixel)
            
            # 检查是否已有该颜色的锁定深度
            if pixel_tuple in self.color_depth_mapping:
                locked_depth = self.color_depth_mapping[pixel_tuple]
                
                # 如果当前深度与锁定深度差异不大，使用锁定深度
                if abs(current_depth - locked_depth) < 20:  # 差异阈值
                    locked_depth_map[non_black_mask][i] = locked_depth
                else:
                    # 差异较大时，更新锁定深度（缓慢更新）
                    new_locked_depth = int(0.7 * locked_depth + 0.3 * current_depth)
                    self.color_depth_mapping[pixel_tuple] = new_locked_depth
                    locked_depth_map[non_black_mask][i] = new_locked_depth
            else:
                # 新颜色，直接锁定当前深度
                self.color_depth_mapping[pixel_tuple] = current_depth
        
        return locked_depth_map
    
    def _apply_histogram_matching_stability(self, depth_map: np.ndarray) -> np.ndarray:
        """
        基于直方图匹配的稳定性算法
        
        Args:
            depth_map: 当前深度图
            
        Returns:
            matched_depth_map: 直方图匹配后的深度图
        """
        if not self.temporal_stability or self.previous_depth_map is None:
            return depth_map
        
        # 计算当前帧和前一帧的深度直方图
        current_hist = np.histogram(depth_map[depth_map > 0], bins=256, range=(0, 256))[0]
        prev_hist = np.histogram(self.previous_depth_map[self.previous_depth_map > 0], bins=256, range=(0, 256))[0]
        
        # 计算累积分布函数
        current_cdf = np.cumsum(current_hist).astype(np.float32)
        prev_cdf = np.cumsum(prev_hist).astype(np.float32)
        
        # 归一化CDF
        current_cdf = current_cdf / current_cdf[-1] if current_cdf[-1] > 0 else current_cdf
        prev_cdf = prev_cdf / prev_cdf[-1] if prev_cdf[-1] > 0 else prev_cdf
        
        # 创建映射表
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # 找到最接近的CDF值
            diff = np.abs(current_cdf[i] - prev_cdf)
            closest_idx = np.argmin(diff)
            mapping[i] = closest_idx
        
        # 应用映射
        matched_depth_map = mapping[depth_map]
        
        return matched_depth_map
    
    def _normalize_depth_globally(self, depth_map: np.ndarray, is_first_frame: bool = False) -> np.ndarray:
        """
        使用全局深度范围进行归一化
        
        Args:
            depth_map: 原始深度图
            is_first_frame: 是否为第一帧
            
        Returns:
            normalized_depth_map: 归一化后的深度图
        """
        if not self.global_normalization:
            # 使用局部归一化
            if depth_map.max() > depth_map.min():
                normalized = ((depth_map - depth_map.min()) / 
                            (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            else:
                normalized = depth_map.copy()
            return normalized
        
        # 更新全局深度范围
        current_min = depth_map.min()
        current_max = depth_map.max()
        
        if is_first_frame or self.global_depth_min is None:
            self.global_depth_min = current_min
            self.global_depth_max = current_max
            # 使用固定的深度范围，避免后续帧的范围变化
            self.fixed_depth_range = (current_min, current_max)
        else:
            # 使用固定的深度范围进行归一化，不再更新范围
            # 这样可以确保所有帧使用相同的深度范围
            if hasattr(self, 'fixed_depth_range'):
                self.global_depth_min, self.global_depth_max = self.fixed_depth_range
            else:
                # 如果没有固定范围，使用保守的更新策略
                self.global_depth_min = min(self.global_depth_min, current_min)
                self.global_depth_max = max(self.global_depth_max, current_max)
                
                # 添加一些缓冲，避免边界值被过度压缩
                range_buffer = (self.global_depth_max - self.global_depth_min) * 0.05
                self.global_depth_min = max(0, self.global_depth_min - range_buffer)
                self.global_depth_max = min(255, self.global_depth_max + range_buffer)
        
        # 使用全局范围归一化
        if self.global_depth_max > self.global_depth_min:
            normalized = ((depth_map - self.global_depth_min) / 
                        (self.global_depth_max - self.global_depth_min) * 255).astype(np.uint8)
        else:
            normalized = depth_map.copy()
        
        return normalized
    
    def reset_temporal_state(self):
        """重置时序状态，用于新视频处理"""
        self.previous_depth_map = None
        self.global_depth_min = None
        self.global_depth_max = None
        self.depth_history = []
        self.previous_image = None
        self.depth_locks = {}
        self.color_depth_mapping = {}
        if hasattr(self, 'depth_stats_history'):
            self.depth_stats_history = []
    
    def _convert_images_to_video(self, images_dir: str, output_path: str, fps: float = 30.0) -> bool:
        """使用FFmpeg将图像序列转换为视频"""
        if not self.ffmpeg_available:
            logger.error("FFmpeg不可用，无法转换视频")
            return False
        
        try:
            # 构建FFmpeg命令
            input_pattern = os.path.join(images_dir, "frame_%06d.jpg").replace('\\', '/')
            # 确保帧率不为0
            if fps is None or fps <= 0:
                fps = 30.0
            
            cmd = [
                'ffmpeg', '-y',  # -y 覆盖输出文件
                '-framerate', str(fps),
                '-start_number', '0',
                '-i', input_pattern,
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path
            ]
            
            logger.info(f"使用FFmpeg转换视频: {output_path}")
            logger.info(f"命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"FFmpeg转换成功: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg转换失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg转换超时")
            return False
        except Exception as e:
            logger.error(f"FFmpeg转换异常: {e}")
            return False
    
    def estimate_depth(self, image: np.ndarray, is_first_frame: bool = False) -> np.ndarray:
        """
        估计单张图像的深度
        
        Args:
            image: 输入图像 (H, W, 3)
            is_first_frame: 是否为第一帧（用于全局归一化）
            
        Returns:
            depth_map: 深度图 (H, W)
        """
        try:
            # 转换图像格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # 检测黑色背景区域（假设黑色背景是抠图后的区域）
            # 直接判断RGB值，使用容差判断
            black_mask = (image_rgb[:, :, 0] < self.black_threshold) & \
                        (image_rgb[:, :, 1] < self.black_threshold) & \
                        (image_rgb[:, :, 2] < self.black_threshold)
            
            if self.depth_mode == "ai":
                # 使用AI模型估计深度
                from PIL import Image
                pil_image = Image.fromarray(image_rgb)
                
                result = self.depth_estimator(pil_image)
                depth_map = np.array(result["depth"])
                
                # 使用全局归一化而不是局部归一化
                depth_map = self._normalize_depth_globally(depth_map, is_first_frame)
                
                # 将黑色背景区域设为0（黑色）
                depth_map[black_mask] = 0
                
            elif self.depth_mode == "color_based":
                # 基于颜色分析计算深度
                color_info = self._analyze_colors(image_rgb)
                depth_map = self._calculate_color_depth(image_rgb, color_info)
                
            elif self.depth_mode == "custom":
                # 基于自定义颜色映射计算深度
                depth_map = self._calculate_custom_color_depth(image_rgb)
                
            else:
                logger.error(f"不支持的深度模式: {self.depth_mode}")
                return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # 应用激进的时序稳定性算法
            # 1. 深度值锁定（最强稳定性）
            depth_map = self._apply_depth_locking_stability(image_rgb, depth_map)
            
            # 2. 直方图匹配（保持深度分布一致性）
            depth_map = self._apply_histogram_matching_stability(depth_map)
            
            # 3. 自适应时序平滑
            depth_map = self._apply_temporal_smoothing(depth_map, alpha=0.2)  # 更强的平滑
            
            # 4. 颜色一致性平滑
            depth_map = self._apply_color_consistent_depth(image_rgb, depth_map)
            
            # 5. 统计特性平滑
            depth_map = self._apply_depth_statistics_smoothing(depth_map)
            
            # 保存当前图像用于下一帧的颜色一致性计算
            self.previous_image = image_rgb.copy()
            
            return depth_map
            
        except Exception as e:
            logger.error(f"深度估计失败: {e}")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    def process_video(self, 
                     input_path: str, 
                     output_path: str,
                     start_frame: int = 0,
                     max_frames: Optional[int] = None,
                     fps: Optional[float] = None) -> bool:
        """
        处理视频文件
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            start_frame: 起始帧
            max_frames: 最大处理帧数
            fps: 输出视频帧率
            
        Returns:
            bool: 处理是否成功
        """
        try:
            # 打开输入视频
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"无法打开输入视频: {input_path}")
                return False
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps is None:
                fps = original_fps
            
            # 如果用户指定了fps为0，使用原视频的fps
            if fps == 0:
                fps = original_fps
                logger.info(f"用户指定fps为0，使用原视频帧率: {fps}fps")
            
            # 只有当fps仍然为0或None时才使用默认值
            if fps is None or fps == 0:
                fps = 30.0
                logger.warning(f"检测到无效帧率，使用默认值30fps")
            
            logger.info(f"视频信息: {width}x{height}, {total_frames}帧, {original_fps:.2f}fps")
            
            # 重置时序状态
            self.reset_temporal_state()
            
            # 设置起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 计算实际处理帧数
            if max_frames is None or max_frames == 0:
                max_frames = total_frames - start_frame
            else:
                max_frames = min(max_frames, total_frames - start_frame)
            
            # 设置输出视频参数
            if self.output_format == "side_by_side":
                output_width = width * 2
                output_height = height
            elif self.output_format == "depth_only":
                output_width = width
                output_height = height
            else:  # overlay
                output_width = width
                output_height = height
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 根据文件扩展名选择合适的编码器
            file_ext = os.path.splitext(output_path)[1].lower()
            
            # 尝试多种编码器创建视频写入器
            if file_ext == '.mp4':
                encoders = [
                    ('mp4v', 'mp4v'),
                    ('H264', 'H264'),
                    ('XVID', 'XVID'),
                ]
            elif file_ext == '.avi':
                encoders = [
                    ('XVID', 'XVID'),
                    ('MJPG', 'MJPG'),
                    ('DIVX', 'DIVX'),
                    ('mp4v', 'mp4v'),
                ]
            elif file_ext == '.mov':
                encoders = [
                    ('mp4v', 'mp4v'),
                    ('MJPG', 'MJPG'),
                ]
            else:
                # 默认尝试所有编码器
                encoders = [
                    ('mp4v', 'mp4v'),
                    ('H264', 'H264'),
                    ('XVID', 'XVID'),
                    ('MJPG', 'MJPG'),
                    ('DIVX', 'DIVX'),
                ]
            
            out = None
            used_encoder = None
            
            for encoder_name, fourcc_code in encoders:
                try:
                    fourcc = cv2.VideoWriter_fourcc(fourcc_code[0], fourcc_code[1], fourcc_code[2], fourcc_code[3])
                    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
                    
                    if out.isOpened():
                        used_encoder = encoder_name
                        logger.info(f"成功使用编码器: {encoder_name}")
                        break
                    else:
                        out.release()
                        logger.warning(f"{encoder_name}编码器失败")
                        
                except Exception as e:
                    logger.warning(f"{encoder_name}编码器异常: {e}")
                    if out:
                        out.release()
            
            if out is None or not out.isOpened():
                logger.error(f"所有编码器都失败，无法创建输出视频: {output_path}")
                logger.info("尝试使用FFmpeg作为备用方案...")
                
                # 尝试使用FFmpeg
                if self._try_ffmpeg_fallback(input_path, output_path, start_frame, max_frames, fps, output_width, output_height):
                    return True
                
                logger.info("FFmpeg也失败，尝试使用图像序列作为最后备用方案...")
                # 重新打开视频文件，因为之前的cap可能已经被消耗
                cap.release()
                cap = cv2.VideoCapture(input_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                return self._process_video_as_images(cap, output_path, max_frames, start_frame, fps, True)
            
            # 处理每一帧
            logger.info("开始处理视频帧...")
            processed_frames = 0
            
            with tqdm(total=max_frames, desc="处理进度") as pbar:
                while processed_frames < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 估计深度，第一帧用于初始化全局深度范围
                    is_first_frame = (processed_frames == 0)
                    depth_map = self.estimate_depth(frame, is_first_frame)
                    
                    # 根据输出格式组合图像
                    output_frame = self._combine_frames(frame, depth_map)
                    
                    # 写入输出视频
                    out.write(output_frame)
                    
                    processed_frames += 1
                    pbar.update(1)
            
            # 释放资源
            cap.release()
            out.release()
            
            logger.info(f"视频处理完成: {processed_frames}帧已处理")
            return True
            
        except Exception as e:
            traceback.print_exc()
            logger.error(f"视频处理失败: {e}")
            return False
    
    def _combine_frames(self, original_frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        根据输出格式组合原始图像和深度图
        
        Args:
            original_frame: 原始图像
            depth_map: 深度图
            
        Returns:
            combined_frame: 组合后的图像
        """
        if self.output_format == "side_by_side":
            # 并排显示
            depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            
            # 确保黑色背景区域在深度图中也保持为黑色
            original_black_mask = (original_frame[:, :, 0] < self.black_threshold) & \
                                (original_frame[:, :, 1] < self.black_threshold) & \
                                (original_frame[:, :, 2] < self.black_threshold)
            depth_colored[original_black_mask] = [0, 0, 0]
            
            combined = np.hstack([original_frame, depth_colored])
            
        elif self.output_format == "depth_only":
            # 只显示深度图，保持黑色背景
            # 创建彩色深度图
            depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            
            # 将深度值为0的区域（黑色背景）保持为黑色
            black_mask = depth_map == 0
            depth_colored[black_mask] = [0, 0, 0]  # 设为黑色
            
            # 确保黑色背景区域在彩色深度图中也保持为黑色
            # 重新检测原始图像的黑色背景，使用RGB容差判断
            original_black_mask = (original_frame[:, :, 0] < self.black_threshold) & \
                                (original_frame[:, :, 1] < self.black_threshold) & \
                                (original_frame[:, :, 2] < self.black_threshold)
            depth_colored[original_black_mask] = [0, 0, 0]
            
            combined = depth_colored
            
        else:  # overlay
            # 叠加显示
            depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            # 创建透明度混合
            alpha = 0.6
            combined = cv2.addWeighted(original_frame, 1-alpha, depth_colored, alpha, 0)
        
        return combined
    
    def process_image(self, input_path: str, output_path: str) -> bool:
        """
        处理单张图像
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            
        Returns:
            bool: 处理是否成功
        """
        try:
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                logger.error(f"无法读取图像: {input_path}")
                return False
            
            # 估计深度
            depth_map = self.estimate_depth(image, is_first_frame=True)
            
            # 根据输出格式组合图像
            output_image = self._combine_frames(image, depth_map)
            
            # 保存结果
            cv2.imwrite(output_path, output_image)
            logger.info(f"图像处理完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return False
    
    def _find_ffmpeg_executable(self) -> str:
        """
        跨平台查找FFmpeg可执行文件
        
        Returns:
            str: FFmpeg可执行文件路径，如果找不到返回None
        """
        import platform
        import shutil
        
        # 首先尝试在PATH中查找
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        
        # 如果PATH中找不到，尝试常见路径
        if platform.system() == 'Windows':
            common_paths = [
                r'C:\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
                r'C:\tools\ffmpeg\bin\ffmpeg.exe',
            ]
        elif platform.system() == 'Darwin':  # macOS
            common_paths = [
                '/usr/local/bin/ffmpeg',
                '/opt/homebrew/bin/ffmpeg',
                '/usr/bin/ffmpeg',
            ]
        else:  # Linux
            common_paths = [
                '/usr/bin/ffmpeg',
                '/usr/local/bin/ffmpeg',
                '/opt/ffmpeg/bin/ffmpeg',
            ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None

    def _try_ffmpeg_fallback(self, input_path: str, output_path: str, start_frame: int, max_frames: int, fps: float, width: int, height: int) -> bool:
        """
        尝试使用FFmpeg作为备用方案处理视频
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            start_frame: 起始帧
            max_frames: 最大处理帧数
            fps: 帧率
            width: 输出宽度
            height: 输出高度
            
        Returns:
            bool: 是否成功
        """
        try:
            import subprocess
            import tempfile
            import shutil
            import platform
            
            # 检查FFmpeg是否可用（跨平台兼容）
            ffmpeg_cmd = self._find_ffmpeg_executable()
            if not ffmpeg_cmd:
                logger.warning("FFmpeg不可用，跳过FFmpeg备用方案")
                return False
            
            # 验证FFmpeg是否正常工作
            try:
                if platform.system() == 'Windows':
                    subprocess.run([ffmpeg_cmd, '-version'], capture_output=True, check=True, shell=True)
                else:
                    subprocess.run([ffmpeg_cmd, '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning(f"FFmpeg路径 {ffmpeg_cmd} 不可用，跳过FFmpeg备用方案")
                return False
            
            logger.info("使用FFmpeg处理视频...")
            
            # 创建临时目录存储处理后的帧
            temp_dir = tempfile.mkdtemp()
            
            try:
                # 处理视频帧并保存到临时目录
                cap = cv2.VideoCapture(input_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                processed_frames = 0
                frame_files = []
                
                with tqdm(total=max_frames, desc="FFmpeg处理进度") as pbar:
                    while processed_frames < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 估计深度，第一帧用于初始化全局深度范围
                        is_first_frame = (processed_frames == 0)
                        depth_map = self.estimate_depth(frame, is_first_frame)
                        
                        # 根据输出格式组合图像
                        output_frame = self._combine_frames(frame, depth_map)
                        
                        # 保存帧到临时文件（跨平台兼容）
                        frame_file = os.path.join(temp_dir, f"frame_{processed_frames:06d}.png")
                        success = cv2.imwrite(frame_file, output_frame)
                        if not success:
                            logger.warning(f"保存帧失败: {frame_file}")
                        frame_files.append(frame_file)
                        
                        processed_frames += 1
                        pbar.update(1)
                
                cap.release()
                
                if processed_frames == 0:
                    logger.error("没有处理任何帧")
                    return False
                
                # 搜索临时目录中的图像文件
                import glob
                image_pattern = os.path.join(temp_dir, "frame_*.png")
                image_files = sorted(glob.glob(image_pattern))
                
                if not image_files:
                    logger.error(f"未找到图像文件: {image_pattern}")
                    return False
                
                logger.info(f"找到 {len(image_files)} 个图像文件")
                
                # 使用concat demuxer方式，更可靠
                # 创建文件列表
                concat_file = os.path.join(temp_dir, "filelist.txt")
                with open(concat_file, 'w', encoding='utf-8') as f:
                    for img_file in image_files:
                        # 使用正斜杠路径，并转义特殊字符
                        img_path = img_file.replace('\\', '/')
                        f.write(f"file '{img_path}'\n")
                        f.write(f"duration {1/fps}\n")
                    # 最后一行不需要duration
                    last_file_path = image_files[-1].replace('\\', '/')
                    f.write(f"file '{last_file_path}'\n")
                
                logger.info(f"创建文件列表: {concat_file}")
                
                # 如果用户指定了fps为0，使用原视频的fps
                if fps == 0:
                    # 重新获取原视频的fps
                    cap_temp = cv2.VideoCapture(input_path)
                    original_fps = cap_temp.get(cv2.CAP_PROP_FPS)
                    cap_temp.release()
                    fps = original_fps
                    logger.info(f"用户指定fps为0，使用原视频帧率: {fps}fps")
                
                # 只有当fps仍然为0或None时才使用默认值
                if fps is None or fps == 0:
                    fps = 30.0
                
                ffmpeg_args = [
                    ffmpeg_cmd, '-y',  # 覆盖输出文件
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file.replace('\\', '/'),
                    '-c:v', 'libx264',  # 使用H.264编码器
                    '-pix_fmt', 'yuv420p',  # 确保兼容性
                    '-crf', '18',  # 高质量
                    '-r', str(fps),  # 设置输出帧率
                    output_path
                ]
                
                logger.info(f"执行FFmpeg命令: {' '.join(ffmpeg_args)}")
                
                # 跨平台兼容的subprocess调用
                if platform.system() == 'Windows':
                    result = subprocess.run(ffmpeg_args, capture_output=True, text=True, shell=True)
                else:
                    result = subprocess.run(ffmpeg_args, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"FFmpeg成功创建视频: {output_path}")
                    return True
                else:
                    logger.error(f"FFmpeg失败: {result.stderr}")
                    return False
                    
            finally:
                # 清理临时文件
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")
                    
        except Exception as e:
            logger.error(f"FFmpeg备用方案失败: {e}")
            return False

    def _process_video_as_images(self, cap, output_path: str, max_frames: int, start_frame: int, fps: float = 30.0, auto_convert: bool = True) -> bool:
        """
        将视频处理为图像序列（备用方案）
        
        Args:
            cap: 视频捕获对象
            output_path: 输出路径（将创建目录）
            max_frames: 最大处理帧数
            start_frame: 起始帧
            
        Returns:
            bool: 处理是否成功
        """
        try:
            # 创建输出目录
            output_dir = output_path.replace('.mp4', '_frames').replace('.avi', '_frames')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            logger.info(f"将视频帧保存到目录: {output_dir}")
            logger.info(f"视频信息 - 总帧数: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}, 当前帧: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}, 最大处理帧数: {max_frames}")
            
            processed_frames = 0
            
            with tqdm(total=max_frames, desc="保存图像序列") as pbar:
                while processed_frames < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"无法读取第 {processed_frames} 帧，可能已到达视频末尾")
                        break
                    
                    # 估计深度，第一帧用于初始化全局深度范围
                    is_first_frame = (processed_frames == 0)
                    depth_map = self.estimate_depth(frame, is_first_frame)
                    
                    # 根据输出格式组合图像
                    output_frame = self._combine_frames(frame, depth_map)
                    
                    # 保存图像
                    frame_filename = os.path.join(output_dir, f"frame_{processed_frames + start_frame:06d}.jpg")
                    cv2.imwrite(frame_filename, output_frame)
                    
                    processed_frames += 1
                    pbar.update(1)
            
            cap.release()
            logger.info(f"图像序列保存完成: {processed_frames}帧已保存到 {output_dir}")
            
            # 如果FFmpeg可用且启用自动转换，尝试转换为视频
            if auto_convert and self.ffmpeg_available and processed_frames > 0:
                logger.info("尝试使用FFmpeg将图像序列转换为视频...")
                video_path = output_path.replace('_frames', '')
                if self._convert_images_to_video(output_dir, video_path, fps):
                    logger.info(f"视频转换成功: {video_path}")
                    # 可选：删除图像序列目录以节省空间
                    import shutil
                    shutil.rmtree(output_dir)
                else:
                    logger.warning("FFmpeg转换失败，保留图像序列")
            
            return True
            
        except Exception as e:
            logger.error(f"图像序列保存失败: {e}")
            cap.release()
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="深度视频转换程序")
    parser.add_argument("input", help="输入视频或图像路径")
    parser.add_argument("output", help="输出路径")
    parser.add_argument("--model", default="Intel/dpt-large", 
                       help="深度估计模型名称 (默认: Intel/dpt-large)")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cuda", "cpu"],
                       help="计算设备 (默认: auto)")
    parser.add_argument("--format", default="side_by_side",
                       choices=["side_by_side", "depth_only", "overlay"],
                       help="输出格式 (默认: side_by_side)")
    parser.add_argument("--depth-mode", default="ai",
                       choices=["ai", "color_based", "custom"],
                       help="深度计算模式: ai(AI模型), color_based(基于颜色), custom(自定义映射) (默认: ai)")
    parser.add_argument("--start-frame", type=int, default=0,
                       help="起始帧 (默认: 0)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="最大处理帧数 (默认: 全部)")
    parser.add_argument("--fps", type=float, default=None,
                       help="输出视频帧率 (默认: 保持原帧率)")
    parser.add_argument("--black-threshold", type=int, default=30,
                       help="黑色背景检测阈值 (默认: 30)")
    parser.add_argument("--force-images", action="store_true",
                       help="强制输出为图像序列而不是视频")
    parser.add_argument("--auto-convert", action="store_true",
                       help="自动使用FFmpeg将图像序列转换为视频")
    
    # 颜色深度相关参数
    parser.add_argument("--color-config", type=str, default=None,
                       help="颜色深度配置文件路径 (JSON格式)")
    parser.add_argument("--custom-color-map", type=str, default=None,
                       help="自定义颜色映射文件路径 (JSON格式)")
    parser.add_argument("--dominant-weight", type=float, default=0.7,
                       help="主色调权重 (默认: 0.7)")
    parser.add_argument("--depth-range", type=str, default="50,200",
                       help="深度范围，格式: min,max (默认: 50,200)")
    parser.add_argument("--similarity-threshold", type=int, default=30,
                       help="颜色相似度阈值 (默认: 30)")
    parser.add_argument("--temporal-stability", action="store_true", default=True,
                       help="启用时序稳定性 (默认: 启用)")
    parser.add_argument("--no-temporal-stability", action="store_true",
                       help="禁用时序稳定性")
    parser.add_argument("--global-normalization", action="store_true", default=True,
                       help="使用全局深度归一化 (默认: 启用)")
    parser.add_argument("--no-global-normalization", action="store_true",
                       help="禁用全局深度归一化")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理颜色深度配置
    color_depth_config = {}
    custom_color_map = {}
    
    # 解析深度范围
    try:
        depth_range = [int(x.strip()) for x in args.depth_range.split(',')]
        if len(depth_range) != 2 or depth_range[0] >= depth_range[1]:
            raise ValueError("深度范围格式错误")
        color_depth_config["depth_range"] = depth_range
    except Exception as e:
        logger.error(f"深度范围解析失败: {e}")
        color_depth_config["depth_range"] = [50, 200]
    
    # 设置其他颜色深度参数
    color_depth_config.update({
        "dominant_weight": args.dominant_weight,
        "similarity_threshold": args.similarity_threshold
    })
    
    # 加载颜色配置文件
    if args.color_config and os.path.exists(args.color_config):
        try:
            with open(args.color_config, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                color_depth_config.update(config_data)
                logger.info(f"已加载颜色配置文件: {args.color_config}")
        except Exception as e:
            logger.error(f"加载颜色配置文件失败: {e}")
    
    # 加载自定义颜色映射文件
    if args.custom_color_map and os.path.exists(args.custom_color_map):
        try:
            with open(args.custom_color_map, 'r', encoding='utf-8') as f:
                custom_color_map = json.load(f)
                logger.info(f"已加载自定义颜色映射文件: {args.custom_color_map}")
        except Exception as e:
            logger.error(f"加载自定义颜色映射文件失败: {e}")
    
    # 处理时序稳定性和全局归一化参数
    temporal_stability = args.temporal_stability and not args.no_temporal_stability
    global_normalization = args.global_normalization and not args.no_global_normalization
    
    # 初始化转换器
    converter = DepthVideoConverter(
        model_name=args.model,
        device=args.device,
        output_format=args.format,
        black_threshold=args.black_threshold,
        depth_mode=args.depth_mode,
        color_depth_config=color_depth_config,
        custom_color_map=custom_color_map,
        temporal_stability=temporal_stability,
        global_normalization=global_normalization
    )
    
    # 判断输入类型
    input_ext = os.path.splitext(args.input)[1].lower()
    
    if input_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
        # 处理视频
        if args.force_images:
            # 强制使用图像序列输出
            cap = cv2.VideoCapture(args.input)
            if not cap.isOpened():
                logger.error(f"无法打开输入视频: {args.input}")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if args.max_frames is None:
                max_frames = total_frames - args.start_frame
            else:
                max_frames = min(args.max_frames, total_frames - args.start_frame)
            
            logger.info(f"强制图像序列输出 - 总帧数: {total_frames}, 起始帧: {args.start_frame}, 处理帧数: {max_frames}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
            success = converter._process_video_as_images(cap, args.output, max_frames, args.start_frame, args.fps or 30.0, args.auto_convert)
        else:
            success = converter.process_video(
                input_path=args.input,
                output_path=args.output,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                fps=args.fps
            )
    elif input_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # 处理图像
        success = converter.process_image(
            input_path=args.input,
            output_path=args.output
        )
    else:
        logger.error(f"不支持的文件格式: {input_ext}")
        return
    
    if success:
        logger.info("转换完成!")
    else:
        logger.error("转换失败!")


if __name__ == "__main__":
    main()
