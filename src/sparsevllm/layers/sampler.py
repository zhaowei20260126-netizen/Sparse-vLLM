import torch
from torch import nn


class Sampler(nn.Module):
    """
    Token 采样器。从模型输出的 logits 中采样得到下一个 token。
    
    支持两种采样模式：
    1. 贪心采样（temperature ≈ 0）：选择概率最高的 token
    2. 概率采样（temperature > 0）：根据概率分布随机采样
    
    使用 Gumbel-Max 技巧实现可编译的多项采样，比 torch.multinomial 更快。

    相比于原始nano-vllm：
        ✓ 支持贪心采样（temperature=0）
        ✓ 数值安全化处理
        ✓ 灵活的混合采样策略
        ✓ 适应更复杂的推理场景（如温度控制、条件生成等）
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        采样 token。
        
        参数：
        - logits: (batch_size, vocab_size) - 模型输出的原始分数
        - temperatures: (batch_size,) - 采样温度，控制采样的随机性
          * temperature ≈ 0：贪心采样（选最高概率）
          * temperature = 1.0：标准概率采样
          * temperature > 1.0：分布更平坦（更随机）
        
        返回：
        - tokens: (batch_size,) - 采样得到的 token ID
        """
        # ==================== 第1步：数据类型转换 ====================
        # 将 logits 转为 float32，确保后续数值运算的精度
        logits = logits.float()
        
        # ==================== 第2步：识别贪心采样 ====================
        # 当 temperature ≤ 1e-10 时，视为贪心采样（接近 0）
        # greedy_mask[i] = True 表示第 i 个序列采用贪心模式
        # greedy_mask[i] = False 表示第 i 个序列采用随机采样模式
        greedy_mask = temperatures <= 1e-10
        
        # ==================== 第3步：安全化温度参数 ====================
        # 问题：贪心模式会后续用 temperature 作为分母，如果 temperature ≈ 0 会导致数值溢出
        # 解决：对贪心模式的序列，把温度设为 1.0，这样不会影响采样结果（因为贪心模式直接选最大值），但避免了除以 0 的风险
        # 贪心结果最后会被单独处理，不会用到这个缩放的 logits
        safe_temperatures = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures) # torch.where(条件, 满足时的值, 不满足时的值)
        # 例如：temperatures = [0.0, 0.7, 0.0, 1.2]
        #      safe_temps   = [1.0, 0.7, 1.0, 1.2]
        
        # ==================== 第4步：温度缩放 logits ====================
        # 用安全的温度值缩放 logits，以调整概率分布的形状
        # unsqueeze(dim=1) 将 (batch_size,) 扩展到 (batch_size, 1) 以便广播
        sampled_logits = logits.div(safe_temperatures.unsqueeze(dim=1)) # 除
        
        # 温度效应说明：
        # - temperature < 1：缩放后的 logits 被放大 → 分布更尖锐 → 贪心倾向增强
        # - temperature = 1：logits 不变 → 分布保持原样 → 标准采样
        # - temperature > 1：缩放后的 logits 被缩小 → 分布更平坦 → 随机性增强
        
        # ==================== 第5步：计算概率分布 ====================
        # 通过 softmax 将缩放后的 logits 转换为概率
        # sum(probs[i]) = 1，每个 token 的概率在 [0, 1] 之间
        probs = torch.softmax(sampled_logits, dim=-1)
        
        # ==================== 第6步：Gumbel-Max 技巧采样（随机采样） ====================
        # 这是一个巧妙的无显式采样的多项分布采样算法，比 torch.multinomial 更快且可编译
        #
        # 数学原理：
        # 从多项分布 probs 中采样，等价于：
        #   sample_idx = argmax_i( log(probs[i]) + Gumbel(0,1) )
        # 其中 Gumbel(0,1) 可以通过指数分布得到：
        #   Gumbel = -log(-log(U))  其中 U ~ Uniform(0,1)
        #   等价于 1 / exponential_distribution(1)
        #
        # 分解步骤：
        # 1. torch.empty_like(probs).exponential_(1)
        #    → 生成指数分布噪声，形状与 probs 相同
        # 2. .clamp_min_(1e-10)
        #    → 钳制噪声下界，避免除以 0 的数值不稳定
        # 3. probs.div_(...)
        #    → 计算 probs / gumbel_noise，等价于应用 Gumbel-Max 技巧
        # 4. .argmax(dim=-1)
        #    → 选择最大值对应的 token 索引
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        
        # ==================== 第7步：贪心采样 ====================
        # 直接从原始 logits 中选择最大值对应的 token
        # 这是最高概率 token，用于 temperature ≈ 0 的情况
        greedy_tokens = logits.argmax(dim=-1)
        
        # ==================== 第8步：混合输出 ====================
        # 根据 greedy_mask 选择输出：
        # - 如果 greedy_mask[i] = True（贪心模式）：使用 greedy_tokens[i]
        # - 如果 greedy_mask[i] = False（随机模式）：使用 sample_tokens[i]
        # 这样可以在同一个 batch 中支持不同的采样策略
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)
