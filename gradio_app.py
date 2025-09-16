#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度视频转换程序 - Gradio网页界面
"""

import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import subprocess
import shutil
import glob
from depth_video_converter import DepthVideoConverter
from images_to_video import ImageSequenceToVideo


def process_image(input_image, model_name, device, output_format, black_threshold, 
                 depth_mode, dominant_weight, depth_range_min, depth_range_max, 
                 similarity_threshold, custom_color_map, temporal_stability, global_normalization):
    """处理图像"""
    try:
        # 保存上传的图像到临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            input_image.save(tmp_file.name)
            input_path = tmp_file.name
        
        # 创建输出文件路径（跨平台兼容）
        output_path = tempfile.mktemp(suffix='.jpg')
        
        # 处理颜色深度配置
        color_depth_config = {}
        custom_map = {}
        
        if depth_mode == "color_based":
            color_depth_config = {
                "dominant_weight": dominant_weight,
                "depth_range": [depth_range_min, depth_range_max],
                "similarity_threshold": similarity_threshold
            }
        elif depth_mode == "custom" and custom_color_map:
            # 解析自定义颜色映射
            try:
                import json
                custom_map = json.loads(custom_color_map)
            except:
                custom_map = {}
        
        # 初始化转换器
        converter = DepthVideoConverter(
            model_name=model_name,
            device=device,
            output_format=output_format,
            black_threshold=black_threshold,
            depth_mode=depth_mode,
            color_depth_config=color_depth_config,
            custom_color_map=custom_map,
            temporal_stability=temporal_stability,
            global_normalization=global_normalization
        )
        
        # 处理图像
        success = converter.process_image(input_path, output_path)
        
        if success and os.path.exists(output_path):
            # 读取处理后的图像
            result_image = cv2.imread(output_path)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # 清理临时文件
            os.unlink(input_path)
            
            return result_image, output_path
        else:
            os.unlink(input_path)
            return None, None
            
    except Exception as e:
        print(f"图像处理错误: {e}")
        return None, None


def process_video(input_video, model_name, device, output_format, black_threshold, 
                 start_frame, max_frames, fps, force_images, auto_convert, 
                 video_codec, video_quality, depth_mode, dominant_weight, 
                 depth_range_min, depth_range_max, similarity_threshold, custom_color_map,
                 temporal_stability, global_normalization):
    """处理视频"""
    try:
        # 保存上传的视频到临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            # 读取视频文件内容
            with open(input_video, 'rb') as f:
                tmp_file.write(f.read())
            input_path = tmp_file.name
        
        # 创建输出文件路径（跨平台兼容）
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # 处理颜色深度配置
        color_depth_config = {}
        custom_map = {}
        
        if depth_mode == "color_based":
            color_depth_config = {
                "dominant_weight": dominant_weight,
                "depth_range": [depth_range_min, depth_range_max],
                "similarity_threshold": similarity_threshold
            }
        elif depth_mode == "custom" and custom_color_map:
            # 解析自定义颜色映射
            try:
                import json
                custom_map = json.loads(custom_color_map)
            except:
                custom_map = {}
        
        # 初始化转换器
        converter = DepthVideoConverter(
            model_name=model_name,
            device=device,
            output_format=output_format,
            black_threshold=black_threshold,
            depth_mode=depth_mode,
            color_depth_config=color_depth_config,
            custom_color_map=custom_map,
            temporal_stability=temporal_stability,
            global_normalization=global_normalization
        )
        
        # 处理视频
        if force_images:
            # 强制图像序列输出
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                os.unlink(input_path)
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if max_frames is None or max_frames == 0:
                max_frames = total_frames - start_frame
            else:
                max_frames = min(max_frames, total_frames - start_frame)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            success = converter._process_video_as_images(
                cap, output_path, max_frames, start_frame, fps or 30.0, auto_convert
            )
        else:
            success = converter.process_video(
                input_path=input_path,
                output_path=output_path,
                start_frame=start_frame,
                max_frames=max_frames,
                fps=fps
            )
        
        # 如果自动转换失败，尝试手动转换
        if auto_convert and force_images and success:
            images_dir = output_path.replace('.mp4', '_frames').replace('.avi', '_frames')
            if os.path.exists(images_dir):
                # 使用独立的图像序列转视频工具
                video_converter = ImageSequenceToVideo()
                final_video_path = output_path.replace('_frames', '')
                convert_success = video_converter.convert_images_to_video(
                    images_dir, final_video_path, fps or 30.0, video_codec, video_quality
                )
                if convert_success:
                    output_path = final_video_path
                    # 清理图像序列目录
                    shutil.rmtree(images_dir)
        
        if success and os.path.exists(output_path):
            # 清理临时文件
            os.unlink(input_path)
            return output_path
        else:
            os.unlink(input_path)
            return None
            
    except Exception as e:
        print(f"视频处理错误: {e}")
        return None


def convert_images_to_video(images_dir, output_path, fps, codec, quality):
    """将图像序列转换为视频"""
    try:
        if not os.path.exists(images_dir):
            return None, "图像目录不存在"
        
        # 检查图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        if not image_files:
            return None, "未找到图像文件"
        
        # 使用图像序列转视频工具
        converter = ImageSequenceToVideo()
        success = converter.convert_images_to_video(
            images_dir, output_path, fps, codec, quality
        )
        
        if success and os.path.exists(output_path):
            return output_path, f"✅ 转换成功，共处理 {len(image_files)} 帧"
        else:
            return None, "❌ 转换失败"
            
    except Exception as e:
        return None, f"❌ 转换错误: {e}"


def create_gradio_interface():
    """创建Gradio界面"""
    
    # 自定义CSS样式
    css = """
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
    }
    .main-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 16px;
        font-weight: bold;
        margin: 15px 0 8px 0;
        color: #2c3e50;
        padding: 8px 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 6px;
        text-align: center;
    }
    .param-group {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #667eea;
    }
    .compact-row {
        display: flex;
        gap: 10px;
        margin-bottom: 8px;
    }
    .compact-row > * {
        flex: 1;
    }
    
    /* 响应式设计 */
    @media (max-width: 1200px) {
        .gradio-container {
            padding: 10px !important;
        }
        .section-title {
            font-size: 14px !important;
            padding: 6px 10px !important;
        }
    }
    
    @media (max-width: 768px) {
        .gradio-container {
            padding: 5px !important;
        }
        .section-title {
            font-size: 12px !important;
            padding: 4px 8px !important;
        }
    }
    
    /* 确保在小屏幕上列布局自适应 */
    @media (max-width: 1024px) {
        .gradio-row {
            flex-direction: column !important;
        }
        .gradio-column {
            width: 100% !important;
            margin-bottom: 20px !important;
        }
    }
    """
    
    with gr.Blocks(css=css, title="深度视频转换器") as demo:
        
        # 标题
        gr.HTML("""
        <div class="main-header">
            <h1>🎥 深度视频转换器</h1>
            <p>将普通视频/图像转换为包含深度信息的视频/图像，支持黑色背景保持</p>
        </div>
        """)
        
        with gr.Row():
            # 左侧：参数设置 (30%)
            with gr.Column(scale=1.2):
                gr.HTML('<div class="section-title">⚙️ 基础设置</div>')
                
                with gr.Group():
                    depth_mode = gr.Dropdown(
                        choices=["ai", "color_based", "custom"],
                        value="ai",
                        label="深度计算模式",
                        info="ai: AI模型 | color_based: 颜色分析 | custom: 自定义映射"
                    )
                    
                    model_name = gr.Dropdown(
                        choices=[
                            "Intel/dpt-large",
                            "Intel/dpt-hybrid-midas", 
                            "facebook/dpt-dinov2-small-kitti"
                        ],
                        value="Intel/dpt-large",
                        label="深度估计模型",
                        info="Intel/dpt-large: 高精度 | Intel/dpt-hybrid-midas: 平衡 | facebook/dpt-dinov2-small-kitti: 快速",
                        visible=True
                    )
                    
                    with gr.Row():
                        device = gr.Dropdown(
                            choices=["auto", "cuda", "cpu"],
                            value="auto",
                            label="计算设备",
                            info="auto: 自动选择 | cuda: GPU加速 | cpu: CPU处理"
                        )
                        output_format = gr.Dropdown(
                            choices=["side_by_side", "depth_only", "overlay"],
                            value="depth_only",
                            label="输出格式",
                            info="side_by_side: 并排显示 | depth_only: 纯深度图 | overlay: 叠加显示"
                        )
                    
                    black_threshold = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=5,
                        label="黑色背景阈值",
                        info="RGB值低于此阈值的像素将被识别为黑色背景"
                    )
                
                gr.HTML('<div class="section-title">🎨 颜色深度设置</div>')
                
                with gr.Group():
                    dominant_weight = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="主色调权重",
                        info="主色调区域的深度权重 (0.1-1.0)",
                        visible=True
                    )
                    
                    with gr.Row():
                        depth_range_min = gr.Slider(
                            minimum=0,
                            maximum=150,
                            value=50,
                            step=10,
                            label="最小深度",
                            info="最小深度值",
                            visible=True
                        )
                        depth_range_max = gr.Slider(
                            minimum=100,
                            maximum=255,
                            value=200,
                            step=10,
                            label="最大深度",
                            info="最大深度值",
                            visible=True
                        )
                    
                    similarity_threshold = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=5,
                        label="颜色相似度阈值",
                        info="颜色相似度判断阈值",
                        visible=True
                    )
                    
                    custom_color_map = gr.Textbox(
                        label="自定义颜色映射 (JSON格式)",
                        placeholder='{"红色": 200, "绿色": 150, "蓝色": 100}',
                        info="自定义颜色深度映射，JSON格式",
                        lines=3,
                        visible=False
                    )
                
                gr.HTML('<div class="section-title">🎬 视频处理参数</div>')
                
                with gr.Group():
                    with gr.Row():
                        start_frame = gr.Number(value=0, label="起始帧", info="从第几帧开始处理")
                        max_frames = gr.Number(value=None, label="最大处理帧数", info="最多处理多少帧（留空处理全部）")
                    
                    fps = gr.Number(value=None, label="输出帧率", info="输出视频的帧率（留空保持原帧率）")
                
                gr.HTML('<div class="section-title">🎬 FFmpeg设置</div>')
                
                with gr.Group():
                    with gr.Row():
                        force_images = gr.Checkbox(value=False, label="强制图像序列输出", info="跳过视频编码器，直接输出图像序列")
                        auto_convert = gr.Checkbox(value=True, label="自动转换为视频", info="使用FFmpeg将图像序列自动转换为视频")
                    
                    with gr.Row():
                        video_codec = gr.Dropdown(
                            choices=["libx264", "libx265", "libvpx-vp9"],
                            value="libx264",
                            label="视频编码器",
                            info="libx264: 兼容性好 | libx265: 压缩率高 | libvpx-vp9: Web优化"
                        )
                        video_quality = gr.Dropdown(
                            choices=["low", "medium", "high", "lossless"],
                            value="medium",
                            label="视频质量",
                            info="low: 快速 | medium: 平衡 | high: 高质量 | lossless: 无损"
                        )
                
                gr.HTML('<div class="section-title">🔧 时序稳定性设置</div>')
                
                with gr.Group():
                    temporal_stability = gr.Checkbox(
                        value=True, 
                        label="启用时序稳定性", 
                        info="减少相邻帧之间的深度值突变，提高视频稳定性"
                    )
                    
                    global_normalization = gr.Checkbox(
                        value=True, 
                        label="使用全局深度归一化", 
                        info="使用整个视频的深度范围进行归一化，确保颜色一致性"
                    )
            
            # 中间：图像和视频处理 (40%)
            with gr.Column(scale=1.6):
                gr.HTML('<div class="section-title">🖼️ 图像处理</div>')
                
                with gr.Row():
                    image_input = gr.Image(
                        label="上传图像",
                        type="pil",
                        height=300
                    )
                    
                    image_output = gr.Image(
                        label="处理结果",
                        height=300
                    )
                
                image_process_btn = gr.Button("🚀 处理图像", variant="primary", size="lg")
                image_download = gr.File(label="下载结果")
                
                gr.HTML('<div class="section-title">🎥 视频处理</div>')
                
                video_input = gr.Video(
                    label="上传视频",
                    height=250
                )
                
                video_process_btn = gr.Button("🚀 处理视频", variant="primary", size="lg")
                
                with gr.Row():
                    video_output = gr.Video(
                        label="处理结果预览",
                        height=300,
                        show_download_button=True
                    )
                    video_download = gr.File(label="下载结果")
            
            # 右侧：图像序列转换和状态 (30%)
            with gr.Column(scale=1.2):
                gr.HTML('<div class="section-title">🔄 图像序列转视频</div>')
                
                with gr.Group():
                    images_dir_input = gr.Textbox(
                        label="图像序列目录路径",
                        placeholder="输入包含图像文件的目录路径",
                        info="目录应包含按顺序命名的图像文件（如frame_000001.jpg）",
                        lines=2
                    )
                    
                    convert_btn = gr.Button("🔄 转换图像序列", variant="secondary", size="lg")
                    
                    with gr.Row():
                        convert_output = gr.Video(
                            label="转换结果预览",
                            height=300,
                            show_download_button=True
                        )
                        convert_download = gr.File(label="下载转换结果")
                
                gr.HTML('<div class="section-title">📊 处理状态</div>')
                
                with gr.Group():
                    status_text = gr.Textbox(
                        label="处理状态",
                        value="等待处理...",
                        interactive=False,
                        lines=4
                    )
                    
                    # 添加一些统计信息
                    gr.HTML("""
                    <div style="margin-top: 10px; padding: 10px; background: #e8f4fd; border-radius: 6px;">
                        <h4 style="margin: 0 0 8px 0; color: #2c3e50;">💡 使用提示</h4>
                    <ul style="margin: 0; padding-left: 20px; font-size: 13px;">
                        <li>GPU加速处理速度更快</li>
                        <li>depth_only格式处理最快</li>
                        <li>强制图像序列可避免编码问题</li>
                        <li>支持批量处理多个文件</li>
                        <li>时序稳定性减少颜色突变</li>
                    </ul>
                    </div>
                    """)
        
        # 控制界面元素显示/隐藏
        def update_ui_visibility(depth_mode):
            if depth_mode == "ai":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            elif depth_mode == "color_based":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
            else:  # custom
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        
        # 事件处理
        def process_image_wrapper(input_image, model_name, device, output_format, black_threshold,
                                depth_mode, dominant_weight, depth_range_min, depth_range_max,
                                similarity_threshold, custom_color_map, temporal_stability, global_normalization):
            if input_image is None:
                return None, None, "请先上传图像"
            
            status_text = "正在处理图像..."
            result_image, output_path = process_image(
                input_image, model_name, device, output_format, black_threshold,
                depth_mode, dominant_weight, depth_range_min, depth_range_max,
                similarity_threshold, custom_color_map, temporal_stability, global_normalization
            )
            
            if result_image is not None:
                status_text = "✅ 图像处理完成"
                return result_image, output_path, status_text
            else:
                status_text = "❌ 图像处理失败"
                return None, None, status_text
        
        def process_video_wrapper(input_video, model_name, device, output_format, 
                                black_threshold, start_frame, max_frames, fps,
                                force_images, auto_convert, video_codec, video_quality,
                                depth_mode, dominant_weight, depth_range_min, depth_range_max,
                                similarity_threshold, custom_color_map, temporal_stability, global_normalization):
            if input_video is None:
                return None, None, "请先上传视频"
            
            status_text = "正在处理视频..."
            output_path = process_video(
                input_video, model_name, device, output_format, 
                black_threshold, start_frame, max_frames, fps,
                force_images, auto_convert, video_codec, video_quality,
                depth_mode, dominant_weight, depth_range_min, depth_range_max,
                similarity_threshold, custom_color_map, temporal_stability, global_normalization
            )
            
            if output_path is not None:
                status_text = "✅ 视频处理完成"
                return output_path, output_path, status_text
            else:
                status_text = "❌ 视频处理失败"
                return None, None, status_text
        
        def convert_images_wrapper(images_dir, fps, video_codec, video_quality):
            if not images_dir or not images_dir.strip():
                return None, None, "请输入图像序列目录路径"
            
            status_text = "正在转换图像序列..."
            output_path = tempfile.mktemp(suffix='.mp4')
            
            result_path, result_status = convert_images_to_video(
                images_dir.strip(), output_path, fps or 30.0, video_codec, video_quality
            )
            
            if result_path is not None:
                return result_path, result_path, result_status
            else:
                return None, None, result_status
        
        # 绑定事件
        # 深度模式变化时更新界面
        depth_mode.change(
            fn=update_ui_visibility,
            inputs=[depth_mode],
            outputs=[model_name, dominant_weight, depth_range_min, depth_range_max, similarity_threshold, custom_color_map]
        )
        
        image_process_btn.click(
            fn=process_image_wrapper,
            inputs=[image_input, model_name, device, output_format, black_threshold,
                   depth_mode, dominant_weight, depth_range_min, depth_range_max,
                   similarity_threshold, custom_color_map, temporal_stability, global_normalization],
            outputs=[image_output, image_download, status_text]
        )
        
        video_process_btn.click(
            fn=process_video_wrapper,
            inputs=[video_input, model_name, device, output_format, black_threshold, 
                   start_frame, max_frames, fps, force_images, auto_convert, 
                   video_codec, video_quality, depth_mode, dominant_weight, 
                   depth_range_min, depth_range_max, similarity_threshold, custom_color_map,
                   temporal_stability, global_normalization],
            outputs=[video_output, video_download, status_text]
        )
        
        convert_btn.click(
            fn=convert_images_wrapper,
            inputs=[images_dir_input, fps, video_codec, video_quality],
            outputs=[convert_output, convert_download, status_text]
        )
        
        # 使用说明
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                <div>
                    <h3 style="margin: 0 0 10px 0; color: #2c3e50;">📖 快速指南</h3>
                    <ul style="margin: 0; padding-left: 20px; font-size: 14px;">
                        <li><strong>AI模式</strong>：使用深度学习模型</li>
                        <li><strong>颜色模式</strong>：基于颜色分析</li>
                        <li><strong>自定义模式</strong>：手动设置颜色深度</li>
                        <li><strong>设备</strong>：有GPU建议选择cuda</li>
                    </ul>
                </div>
                <div>
                    <h3 style="margin: 0 0 10px 0; color: #2c3e50;">🎨 颜色深度</h3>
                    <ul style="margin: 0; padding-left: 20px; font-size: 14px;">
                        <li>✅ 保持黑色区域不变</li>
                        <li>✅ 主色调区域深度更高</li>
                        <li>✅ 支持自定义颜色映射</li>
                        <li>✅ 实时调整参数</li>
                    </ul>
                </div>
                <div>
                    <h3 style="margin: 0 0 10px 0; color: #2c3e50;">🎯 特色功能</h3>
                    <ul style="margin: 0; padding-left: 20px; font-size: 14px;">
                        <li>✅ GPU加速处理</li>
                        <li>✅ 在线视频预览</li>
                        <li>✅ 多种编码器支持</li>
                        <li>✅ 批量处理支持</li>
                        <li>✅ 时序稳定性优化</li>
                    </ul>
                </div>
            </div>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    # 创建并启动Gradio界面
    demo = create_gradio_interface()
    
    print("🚀 启动深度视频转换器网页界面...")
    print("📱 界面将在浏览器中自动打开")
    print("🌐 如果没有自动打开，请访问显示的URL")
    
    # 启动界面
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        share=False,            # 是否创建公共链接
        debug=False,            # 调试模式
        inbrowser=True,
        show_error=True         # 显示错误信息
    )
