#!/usr/bin/env python3
"""
Llama模型使用示例
展示如何使用集成了Flash Attention的Llama模型
"""

import torch
from models.basic_mntp import create_llama_model, get_model_size

def main():
    print("=== Llama模型使用示例 ===\n")
    
    # 创建一个小型模型用于演示
    print("1. 创建模型...")
    model = create_llama_model(
        vocab_size=1000,      # 小词汇表
        dim=512,              # 小维度
        n_layers=6,           # 6层
        n_heads=8,            # 8个注意力头
        n_kv_heads=4,         # 4个KV头（GQA）
        max_seq_len=1024,     # 最大序列长度
        use_flash_attn=True   # 使用Flash Attention
    )
    
    print("✓ 模型创建完成")
    
    # 获取模型信息
    print("\n2. 模型信息:")
    model_info = get_model_size(model)
    print(f"   总参数数量: {model_info['total_parameters']:,}")
    print(f"   可训练参数: {model_info['trainable_parameters']:,}")
    print(f"   参数大小: {model_info['parameter_size_mb']:.2f} MB")
    print(f"   使用Flash Attention: {model_info['use_flash_attn']}")
    print(f"   Flash Attention可用: {model_info['flash_attn_available']}")
    
    # 创建示例输入
    print("\n3. 创建示例输入...")
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    
    # 创建随机token
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"   输入形状: {tokens.shape}")
    
    # 前向传播
    print("\n4. 前向传播...")
    model.eval()
    with torch.no_grad():
        try:
            logits = model(tokens)
            print(f"   输出形状: {logits.shape}")
            print(f"   输出范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            print("✓ 前向传播成功")
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
    
    # 性能对比（如果Flash Attention可用）
    if model_info['flash_attn_available']:
        print("\n5. Flash Attention vs 标准注意力性能对比:")
        
        # 创建不使用Flash Attention的模型
        model_standard = create_llama_model(
            vocab_size=1000, dim=512, n_layers=6, n_heads=8,
            n_kv_heads=4, max_seq_len=1024, use_flash_attn=False
        )
        
        # 测试不同序列长度
        seq_lengths = [64, 128, 256, 512]
        
        for seq_len in seq_lengths:
            tokens = torch.randint(0, 1000, (1, seq_len))
            
            # 测试Flash Attention
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
            
            with torch.no_grad():
                _ = model(tokens)
            
            if torch.cuda.is_available():
                end_time.record()
                torch.cuda.synchronize()
                flash_time = start_time.elapsed_time(end_time)
            else:
                import time
                start = time.time()
                with torch.no_grad():
                    _ = model(tokens)
                flash_time = (time.time() - start) * 1000  # 转换为毫秒
            
            print(f"   序列长度 {seq_len:3d}: Flash Attention ~{flash_time:.1f}ms")
    
    print("\n=== 示例完成 ===")

if __name__ == "__main__":
    main()
