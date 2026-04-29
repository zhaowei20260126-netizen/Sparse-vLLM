### **AttentionPredictor：Temporal Patterns Matter for KV Cache Compression**

> NeurIPS 2025

#### **背景**

长上下文任务（如 CoT 推理）是 LLM 的发展趋势，但KVcache占用的显存太大，远超模型参数本身，导致显存和计算压力大，限制推理效率。

#### **现有方法问题**

- 近期的研究方法通过对注意力分数进行**静态建模**（基于启发式规则或者历史统计数据）来识别关键的KV token。但是它们忽视了**注意力分数中的时间模式（temporal patterns）**，因此很难准确判定关键token，进而导致LLM性能下降。

> 所谓静态建模就是**简单的累加**，或者**“用一套固定的规则（如只看最近的词）来硬套所有情况”**。通常包含以下两种主流情况：
>
> 1. 历史累积型：代表算法就是H2O，根据一个token在过去的步骤中被关注了多少次，来决定他是否重要。
> 2. 固定规则型：代表算法是StreamingLLM，只保留最开头的几个token的kv（Attention sink）、以及最近的n个token（Sliding Window）。之所以固定，是因为他保留哪个位置的token是写死在代码里的规则，与输入内容无关。

- 最近SeerAttention  和 Attention-Gate （学习型方法）使用可学习模块来建模**动态模式**并检索关键 Token。但是它们仅仅表示key或者隐藏状态，**不是直接对注意力分数分布进行建模**，因此在压缩KVcahe后表现出有限的准确性。

#### 洞察

* 注意力分数并非杂乱无章，而是具有稳定的**时间模式（Temporal Patterns）**，具体表现为：重访问（回头看）、顺序移动（往下读）、周期性（有节奏）。

* 这使得 KV Cache 压缩本质上可以转化为一个**二维时间序列预测问题**。

  <img src="C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126201506990.png" alt="image-20251126201506990" style="zoom:50%;" />

#### **论文解决方法**

提出**AttentionPredictor**（首个基于学习的方法），可以直接预测注意力模式（注意力分数分布），用以KVcache的压缩和关键token的识别。同时，论文提出一个**跨Token的关键cache预取框架**（系统级优化），利用预测的时间差提前加载数据，掩盖了预测和传输带来的延迟。

> AttentionPredictor：NeurIPS 2025：NeurIPS 2025具体原理就是：提前训练一个轻量级、通用的**CNN模型**，用来动态捕捉**时空模式（spatiotemporal patterns）**并预测下一个token的注意力分数。
>
> 该CNN模型也就是**预测模型**，所有Transformer层之间共享一个CNN模型，且该模型占用的显存微乎其微。

#### 主要成果

*   实现了 **13倍** 压缩，性能无损。
*   在缓存卸载场景下加速 **5.6倍**。

<hr>

#### 一、相关工作

1. **高效LLM推理**

   - **投机解码**使用小的草稿模型预测多个未来的Token，并由目标模型并行验证。
   - 论文中的方法则是预测下一个Token的注意力分数来估计KVcache的压缩策略。本文针对的是解码阶段的显存优化，和现有的预填充阶段优化（如前缀缓存）等技术是兼容的。

2. **KV Cache压缩**

   许多方法利用注意力分数具有稀疏性这一发现来压缩KV Cache。

   - **缓存驱逐（Cache Eviction）**：使用**启发式**手段来识别关键KV，同时驱逐相关性较低的KV。具体方法如下：

     - **StreamingLLM:**  观察到decode过程中最早的token具有高注意力分数，所以它只保留初始token和最近少量token。
     - **H2O:** 累积所有token的历史注意力分数，低分的将会被永久驱逐。但由于前几个 Token 的分数累积过于频繁，导致存在累积误差。
     - **SnapKV:** 在最近窗口内累积注意力分数作为估计，并在预填充阶段执行一次性过滤。
     - **MInference**  和 **FlexPrefill： **归纳定义了几种注意力模式，并确定每个头的压缩策略以加速预填充阶段。

     > 所有这些方法都是**基于启发式的**，只能捕捉统计规律。难以准确捕捉注意力分数中动态的**时间模式**。本文通过学习**时间模式**来实现动态预测。

   - **缓存检索（Cache Retrieval）**：检索当前查询时的关键token的kv，只加载他们到计算单元

     - **Quest**、**InfLLM**和 **PQCache** 等方法通过估计压缩后的 Key 表示上的注意力权重，来检索关键 Token，压缩会引入检索误差。特别是**Quest对页面大小敏感**，当页面较大且预算较小时，准确率会下降。
     - 论文中的方法通过轻量级的时间预测器直接预测注意力权重，无需对Key进行有损压缩。检索更准。

   - **基于训练的方法（Training-based Approaches）**：

     - **MoBA** 扩展了缓存检索方法，在训练期间集成了稀疏注意力。**SeerAttention** 和 **NSA** 训练线性模型来更准确地编码和检索关键块。这些方法通常需要**为每一层训练一个单独的模型**，限制可扩展性和泛化能力。NSA还依赖于对LLM本身的**微调**来获得最佳结果。
     - 本文方法采用一个单一的**即插即用**（Plug-and-Play）模块，可统一应用于统一LLM内的所有层和头，且**不需要任何额外微调**，适用范围广。

   - **KV Cache 跨层预取（KV Cache Cross-layer Prefetching）**：将Cache卸载到CPU，跨层预取到GPU

     - **InfiniGen** 结合了缓存检索和预取，通过近似下一层的注意力分数来加载关键 Cache。但是，其估计时间随着序列的增长而显著增加，单层的推理时间不足以覆盖这一开销。

   > AttentionPredictor： 通过**动态预测、无损检索、轻量共享模型、跨 Token 预取**，精准打击了上述所有痛点。

#### 二.  预备知识及动机

1. **预备知识**

   - $Q_t \in \mathbb{R}^{1 \times d}$ 和 $K \in \mathbb{R}^{t \times d}$ 分别表示用于生成第 **$t$** 个 Token 的**查询张量**和**键张量**，第 $t$ 步的注意力分数计算：$$ A_t = \text{Softmax}\left( \frac{1}{\sqrt{d}} Q_t K^\top \right) = [a_{t,1}, a_{t,2}, \dots, a_{t,t}] \in \mathbb{R}^{1 \times t} $$

   - 基于稀疏性的 KV Cache 压缩旨在找到一个预算为 $B$ 的Key子集，该子集保留了最重要的注意力值。具体来说，键索引的子集为 $S = \{s_1, s_2, \dots, s_B\}$，其中每个 $s_j$ 是总共 $t$ 个 Token 中的索引。将**注意力恢复率**定义为：

     ![image-20251125223635877](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251125223635877.png)

     > 上面的公式就是下面的表达（其中由于注意力分数经过softmax操作，所以理论上分母永远等于1，即原本所有Token的分数总和始终等于）
     >
     > ![image-20251125223810725](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251125223810725.png)
     >
     > 这公式反映了压缩后保留的信息量。较高的恢复率 $R_{rec}$ 表示 KV Cache 压缩造成的信息损失较少。因此，**KV Cache 压缩的目标**可以表述为找到使 $R_{rec}$ 最大化的索引集 $S$。
     >
   
   **2.问题定义**
   
   - 将 KV Cache 压缩转化为一个 **2维 时间序列预测问题**。也就是注意力分数预测，根据分数挑选重要的token对应的kv，进行一个cache压缩。空间轴（横轴）对应上下文中的 Token 位置，二维中时间轴（纵轴）对应解码步骤。（如下图）。
   
     ![image-20251126204343229](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126204343229.png)
   
     > 该时序序列预测问题核心思路，就是准确预测出下一步（t+1）模型会关注哪些token，就保留这些位置的Cache，其他都可丢弃。这是一个真正的**利用小模型CNN来预测关键token**的思路。
   
   3**. 注意力时间模式（ Attention Temporal Patterns）**
   
   - 注意力模式：论文中作者发现，在LLM生成过程中，注意力分数的分布在时间轴和空间轴上呈现三种稳定的集合形状，也就是三大核心模式
   
     1. **重访问模式 (Re-access)**： 在2d图中表现为竖直的线条。含义是说模型不管生成到哪一步，都要回头看某些特定的固定Token（这些通常是System Prompt或开头的关键名词）
     2. **顺序模式 (Sequential)**：在2d时空图中表现为向右下角移动的对角斜线。含义就是说模型它和人类一样对文字的阅读是逐字阅读以生成下一步词的。就比如读完第5个词，下一步必然读第6个词。这是最自然的语言处理逻辑。
     3. **周期性模式 (Seasonal)**：在2d时空图中像斑马线一样的周期性条纹。含义是说某些关注点会周期性地复现。
   
     > 这些模式都可以用一个“平移不变性”公式概括：
     > $$ a_{t,i} \approx a_{t+\delta_t, i+\delta_i} $$（其中 $\delta_t$ 和 $\delta_i$ 分别表示时间和空间的偏移量。）
     > 意思是：**明天的注意力分布 $\approx$ 把今天的分布图按某种规律平移一下。**
   
     ![image-20251126091938601](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126091938601.png)
   
     > 图2：三种可预测的时间注意力模式的可视化
   
   - 那么为什么注意力会遵循这些模式呢？
   
     这是LLM的固有属性所导致的，论文发现主要是基于下面两个原因
   
     1. **Query 的自相似性 (High Query Self-similarity)**：在LLM 生成时，第 𝑡 步的 Query 向量 (𝑞𝑡) 和第 𝑡+1 步的 Query 向量 (𝑞𝑡+1) 长得非常像（余弦相似度平均高达 0.87，下图为论文实验的相邻时间步 Query 向量的余弦相似度热力）。既然“搜索的关词”（Query）几乎没变，那么“搜索结果”（Attention Score）自然也不会剧烈突变。这就解释了为什么注意力分布在时间步上具有连续性和稳定性。
   
        ![image-20251126090334756](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126090334756.png)
   
     2. **位置编码的作用(RoPE)：**现在LLM是有旋转位置编码，注意力分数高度依赖于**相对位置** (j-i)，当相对位置固定时（比如 j-i=1，即始终关注前一个词），Attention 分数就会保持稳定，这就从数学上强制形成了**顺序模式（斜线）**。同时当 $j-i$ 变化配合余弦函数性质时，就会形成**周期性波纹（周期模式）**
   
   - *注意：*这三种基础模式并非静止的（**动态模式**），甚至在单个生成序列中也会表现出动态演变。随着解码过程的进行，模式可能会发生**偏移**、消退，如图2所示。
   
     - **原因**：query会偏移，即$虽然q_t$ 和 $q_{t+1}$ 很像（局部稳定），但是，$q_t$ 和 $q_{t+1000}$ 可能就**完全不像**了。因为 $query$变了（漂移了），它和旧的 $K$（几十步之前存下的 Key）相乘得到的注意力分数分布就会变化很大。
     - 这种动态性要求预测器不经能够还原稳定的结构（如重访问、顺序模式），还要适应**模式的偏移**

     > 由于这些模式是LLM固有特征，和输入数据没关，因此可以为每个任务单独训练模型，一个通用的预测器就可以服务所有层、头和数据集，这就解释了模型为什么能做到轻量和动向。

   因为 **Query 的稳定性** 提供了时间上的连续性，而 **RoPE 的相对位置特性** 提供了空间上的几何结构，所以 Attention Score 本质上是一个**有规律可循的 2D 时空信号**。这完美地证明了使用 **CNN（卷积神经网络）** 来捕捉这些几何特征并预测未来是科学且合理的。

#### 三、方法

具体分为注意力预测小模型（AttentionPredictor：NeurIPS 2025）与跨令牌 KV 缓存预取框架两部分（如下图），前者通过动态时空建模精准筛选关键 token，后者通过异步加载机制隐藏计算与传输延迟

![image-20251126101456615](C:\Users\Zhaowei\AppData\Roaming\Typora\typora-user-images\image-20251126101456615.png)

>  KV Cache 压缩方法 AttentionPredictor：NeurIPS 2025 和跨 token 预取框架。（a）AttentionPredictor：NeurIPS 2025 将历史注意力分数建模为时空序列，并借助预训练模型预测下一步的注意力。为了提升效率，在每个 decoding 步骤中，历史注意力分数会以压缩形式进行更新。（b）跨 token 预取框架。在 LLM 推理过程中，异步评估关键 token，并为下一个 token 获取 KV，从而有效加速解码阶段。

- **AttentionPredictor：NeurIPS 2025具体流程**

  假设当前模型正在处理第 $t$ 个 Token，算法需要为第 $t+1$ 个 Token 做准备。

  * **步骤 1：输入准备 (Input Preparation)**

    *   获取当前第 $t$ 步的真实注意力分数 $A_t$。
    *   **分块压缩 (Block-wise Compression)：** 原始的 $A_t$ 长度可能有一万多，直接处理太慢。算法对其进行 **Max-Pooling（最大池化）**。
        *   假设块大小 $b=16$，就是每 16 个分数取一个最大值。
        *   得到压缩后的向量 $A^{comp}_t$。

  * **步骤 2：更新历史图谱 (Update History Buffer)**

    *   维护一个历史队列 $A_H$（一个二维矩阵，存了最近 $H$ 步的压缩注意力）。
    *   将刚刚算好的 $A^{comp}_t$ 塞进队列尾部，挤掉最旧的一步。
    *   **关键点：分布误差校准 (Calibration)**
        *   因为平时我们在用稀疏 Cache 跑，算出来的 $A_t$ 其实是不完整的（有误差）。
        *   为了防止误差滚雪球，算法设定每隔 $M$ 步（比如 5 步），强制让 GPU 跑一次**全量 Attention**。
        *   用这次全量的“真值”来刷新 $A_H$，确保预测器的“视力”没有偏差。

  * **步骤 3：CNN 预测 (Prediction)**

    * 将更新好的历史图谱 $A_H$ 喂给 **CNN 模型**。

    * CNN 识别图谱中的“竖线”（重访问）、“斜线”（顺序读取）等模式。

    * **输出：** 预测出的下一步（$t+1$）的注意力热力图 $\hat{A}_{t+1}$。

      > 本文采用的CNN架构：
      >
      > ![image-20251126210325960](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126210325960.png)

  * **步骤 4：Top-K 决策 (Selection)**

    *   根据显存预算 $B$，从预测结果 $\hat{A}_{t+1}$ 中选出得分最高的 $K$ 个块（Block）。
    *   将这些块的 ID 映射回具体的 Token 索引集合 $S$。
    *   **结果：** 集合 $S$ 就是第 $t+1$ 步必须在显存里准备好的数据。

  ![image-20251126150415547](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126150415547.png)

  

- **Cross-token Prefetch System:** 利用流水线并行的思想，利用 **Token t 的计算时间** 来覆盖 **Token t+1 kv 的 I/O 传输时间**。

![image-20251126163610314](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126163610314.png)

> 以前的旧方法是跨层预取，在在计算第 1 层（Layer 1）的时候，去取第 2 层（Layer 2）的KV数据。但由于LLM计算一层时间极短，意味着预测和数据传输必须要在该极端的时间内完成，可能会来不及运行复杂的预测模型。
>
> 本文采用跨token预取，在计算**第 t 个 Token**（包含所有层）的时候，去取**第 t+1个 Token** 所需要的kv数据。留给CNN这种准确的模型预测时间更长，掩盖延迟（只要传输时间小于生成一个token的时间，io延迟就是0）



#### 四.实验

1.**实验设置**

- **模型：**
  - **LLaMA-3.1-8B-Instruct (128K context)：** 主流的长窗口模型，用来测试超长文本。
  - **LongChat-v1.5-7b-32k：** 另一个经典的长文本模型。
- **任务:**
  - **综合能力：** LongBench（涵盖摘要、问答、代码等6个任务）,一个广泛应用于长上下文LLM推理的基准测试。
  - **超长文本：** InfiniteBench（平均 124K 长度，极限测试）。
  - **长逻辑推理 (Chain-of-Thought)：** **AIME**（高难度数学竞赛题）和 **GSM8K**。测试模型在“输出”很长时的表现。
  - **记忆力测试：** Needle In A Haystack（大海捞针）
- **基线：**
  - **驱逐派 (Eviction)：** H2O, StreamingLLM, SnapKV（代表静态规则）。
  - **检索派 (Retrieval)：** Quest（代表有损压缩检索）。
  - **学习派 (Learning)：** SeerAttention（代表上一代学习型方法）。

- **硬件：** NVIDIA A800 (80GB)
- **超参数设置：**
  - 历史步数H： 64
  - 块大小b：16
  - 校准步数M：5

> 仿照Quest的做法，对LLM的前两层不应用论文中的方法或其他算法，保持全量cache
>
> 遵循 H2O 和 StreamingLLM 的设置，我们将预算平均分配给前缀（Prefix）和局部（Local）Token，各分配 64 个 Token（即保留开头和最近的 Token）。剩余的 KV 预算分配给**中间 Token**，这部分由预测模型决定。

**2.主实验结果**

 AttentionPredictor 在多种任务上均为目前最强（SOTA）的稀疏 Attention 方法，性能最接近全量 Cache。

- **长上下文综合能力 (LongBench)**

   在 LLaMA-3.1-8B 和 LongChat-7B 上，将 KV Cache 压缩至 1024、2048、4096 个 Token

![image-20251126184402415](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126184402415.png)

- **长链式推理能力 (CoT Reasoning)**

  使用 QwQ-32B 模型解决高难度数学题（AIME）。特点是**输入短，但输出极长**（推理过程长达 20K+ Token）。

![image-20251126184824146](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126184824146.png)

**3.Needle In A Haystack (大海捞针测试)**

在极端稀疏度（1/64）下，只有 AttentionPredictor：NeurIPS 2025 能做到 100% 不丢包

![image-20251126185105274](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126185105274.png)

4.  **Efficiency (效率与延迟)**

   通过“跨 Token 预取”技术，在缓存卸载（Offloading）场景下实现了 **5.6倍** 加速，且延迟不随长度增加而线性增长。

   > 实验设置： 显存放不下，KV Cache 放在 CPU 内存，计算时搬运

   <img src="C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126185511820.png" alt="image-20251126185511820" style="zoom:67%;" />

   > *   **FlashAttention2 (灰色柱)：** 随着 Context 变长（4K -> 32K），延迟线性飙升（因为搬运数据量变大）。
   > *   **H2O (浅蓝柱)：** 虽然只搬运部分数据，但因为没有预取优化，依然有 I/O 开销。
   > *   **AttentionPredictor：NeurIPS 2025 (深绿柱)：** 延迟几乎是一条**平线**。

   <img src="C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126185907109.png" alt="image-20251126185907109" style="zoom:67%;" />

   此外根据table2可知只要 Wait for Main LLM 的时间大于 Predict + Transfer，用户感知的额外延迟就是 0。

5. **Prediction Accuracy (预测准确率)**

   - 恢复率（$$R_{rec}$$，即保留下来的 Attention 分数占总分数的比例）更高

     > CNN 预测出来的 Top-K，比 Quest 估算出来的 Top-K 更接近真实情况。

   <img src="C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126190202735.png" alt="image-20251126190202735" style="zoom:70%;" />

   - 跨任务泛化能力强。在 LongBench 训练的预测模型，直接拿去预测其他完全不同的任务，准确率依然保持在 **95% 以上**。

     <img src="C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126190757779.png" alt="image-20251126190757779" style="zoom:80%;" />

   **6. Ablation Study (消融实验)**

   通过调整 $b$ 的大小（8, 16, 32, 64），发现b为16时性价比最高。同时可以发现对quest而言，随着b变大（页大小变大），其准确率下降很快，说明其对粗粒度信息的容忍度降低。

   ![image-20251126191400066](C:\Users\Zhaowei\Desktop\论文\AttentionPredictor：NeurIPS 2025\1. Introduction 详细翻译.assets\image-20251126191400066.png)



#### 附：CNN模型训练过程

论文采用的是**监督学习（Supervised Learning）**，训练过程如下：

*   **数据收集：** 作者在预训练好的 LLM（如 LLaMA-3）上跑一遍全缓存（Full Cache）的推理，记录下真实的注意力分数（Ground Truth Attention Scores）。
*   **构建样本：**
    *   **输入 ($X$)：** 截取过去 $H$ 步的注意力分数图（经过 Max-pooling 压缩）。
    *   **标签 ($Y$)：** 下一步（第 $t+1$ 步）真实的注意力分数向量。
*   **损失函数：** 使用 **MSE (均方误差)** 损失函数，计算预测分数和真实分数之间的差异。
*   **训练特点：**
    *   **数据高效：** 只用了大约 3% 的数据（来自 LongBench 的一小部分）进行训练。
    *   **泛化性强：** 训练好的这个小模型可以直接迁移到其他数据集，甚至不需要微调 LLM 本身。





