import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
from typing import Tuple, Union

# 尝试导入flash_attn，如果不可用则使用标准注意力
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, falling back to standard attention")


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LlamaAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None, qkv_proj_bias: bool = False, out_proj_bias: bool = False, use_flash_attn: bool = True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=qkv_proj_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=qkv_proj_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=qkv_proj_bias)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=out_proj_bias)
        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        
        # 计算Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头格式
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        # 应用RoPE
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        
        # 重复K和V用于分组查询注意力
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        
        if self.use_flash_attn:
            # 使用Flash Attention
            # Flash Attention需要 (batch_size, seq_len, num_heads, head_dim) 格式
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=0.0,  # 训练时可以设置dropout
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True  # 因果掩码
            )
            # Flash Attention输出已经是 (bsz, seqlen, n_heads, head_dim) 格式
            attn_output = attn_output.reshape(bsz, seqlen, -1)
        else:
            # 使用标准注意力计算
            # 转置为 (bsz, n_heads, seqlen, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # 计算注意力分数
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                scores = scores + mask
            
            # 应用softmax
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            
            # 应用注意力权重
            attn_output = torch.matmul(attn_weights, v)
            
            # 转置回 (bsz, seqlen, n_heads, head_dim)
            attn_output = attn_output.transpose(1, 2).contiguous()
            
            # 重塑
            attn_output = attn_output.reshape(bsz, seqlen, -1)
        
        return self.out_proj(attn_output)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """应用旋转位置编码"""
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    x_ = torch.view_as_complex(x_)
    freqs_cis = torch.view_as_complex(freqs_cis)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """预计算旋转位置编码的频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class LlamaMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        # 确保hidden_dim是multiple_of的倍数
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None, mlp_hidden_dim: int = None, norm_eps: float = 1e-5, use_flash_attn: bool = True):
        super().__init__()
        self.attention = LlamaAttention(dim, n_heads, n_kv_heads, use_flash_attn=use_flash_attn)
        self.feed_forward = LlamaMLP(dim, mlp_hidden_dim or 4 * dim)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 预归一化注意力
        norm_x = self.attention_norm(x)
        h = x + self.attention(norm_x, freqs_cis, mask)
        
        # 预归一化前馈网络
        norm_h = self.ffn_norm(h)
        out = h + self.feed_forward(norm_h)
        
        return out


class LlamaModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int, 
                 n_kv_heads: int = None, mlp_hidden_dim: int = None, 
                 max_seq_len: int = 2048, norm_eps: float = 1e-5, use_flash_attn: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.use_flash_attn = use_flash_attn
        
        # 词嵌入层
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Transformer层
        self.layers = nn.ModuleList([
            LlamaBlock(dim, n_heads, n_kv_heads, mlp_hidden_dim, norm_eps, use_flash_attn)
            for _ in range(n_layers)
        ])
        
        # 输出层归一化
        self.norm = RMSNorm(dim, eps=norm_eps)
        
        # 输出投影层
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # 预计算位置编码
        self.register_buffer('freqs_cis', precompute_freqs_cis(dim, max_seq_len))
        
    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # 获取对应的位置编码
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        
        # 创建因果掩码
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float('-inf'), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen, seqlen)
        
        # 通过所有transformer层
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        
        # 最终归一化
        h = self.norm(h)
        
        # 输出投影
        logits = self.output(h)
        
        return logits


def create_llama_model(vocab_size: int = 32000, dim: int = 4096, n_layers: int = 32, 
                     n_heads: int = 32, n_kv_heads: int = None, max_seq_len: int = 2048,
                     use_flash_attn: bool = True) -> LlamaModel:
    """
    便捷函数创建Llama模型
    
    Args:
        vocab_size: 词汇表大小
        dim: 模型维度
        n_layers: 层数
        n_heads: 注意力头数
        n_kv_heads: KV头数（用于GQA，默认为n_heads）
        max_seq_len: 最大序列长度
        use_flash_attn: 是否使用Flash Attention
    
    Returns:
        LlamaModel实例
    """
    return LlamaModel(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=max_seq_len,
        use_flash_attn=use_flash_attn
    )


def get_model_size(model: LlamaModel) -> dict:
    """
    获取模型大小信息
    
    Args:
        model: LlamaModel实例
    
    Returns:
        包含参数数量和内存使用信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算内存使用（以MB为单位）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_size_mb': param_size,
        'use_flash_attn': model.use_flash_attn,
        'flash_attn_available': FLASH_ATTN_AVAILABLE
    }