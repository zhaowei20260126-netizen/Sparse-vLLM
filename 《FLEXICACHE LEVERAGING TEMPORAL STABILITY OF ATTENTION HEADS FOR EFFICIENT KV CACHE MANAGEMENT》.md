### 《FLEXICACHE: LEVERAGING TEMPORAL STABILITY OF ATTENTION HEADS FOR EFFICIENT KV CACHE MANAGEMENT》

> MLsys 2026
>
> 本文针对长上下文推理场景，基于观察：一些关键token的**时间稳定性**在不同的kv头之间存在显著差异，有些头持续关注相同的 token，而有些头则频繁变化。提出了**FlexiCache**，即一个**分层的 KV 缓存管理系统**，在GPU内存中保留不稳定头的**所有KV缓存页**，而对于稳定头，仅保留**top-k个页**在gpu上，其余offload到cpu内存里。
>
> **FlexiCache 在 vLLM 之上实现**，将长上下文请求的 GPU 内存占用减少高达 70%，将离线服务吞吐量提高 1.38–1.55 倍，并将在线 token 延迟降低 1.6–2.1 倍，同时在长上下文、长生成场景下保持准确性。



### 1、研究背景与动机

*   **长上下文与长生成的需求增长**：LLMs 被越来越广泛地应用于代码生成、长文写作等任务，这些任务不仅具有极长的输入上下文，还需要生成极长的输出。

*   **KV Cache 带来的内存瓶颈**：在 Decode 阶段，KV Cache 的大小随着上下文和生成长度的增加呈线性增长。庞大的 KV Cache 会迅速耗尽 GPU 内存，严重限制了系统的批处理大小（Batch Size），从而导致吞吐量低下。

*   **现有稀疏注意力方法的局限性**：

    *   **永久丢弃策略（如 SnapKV, StreamingLLM）**：通过注意力分数丢弃被认为不重要的 KV Cache。虽然节省了内存，但在**长生成任务**中表现不佳，因为被丢弃的 token 在后续生成中可能重新变得重要，永久丢弃会导致模型精度下降。
    *   **全保留+动态选择策略（如 Quest）**：在每一小步重新计算 Top-K 的 KV 页面进行注意力计算，虽然保住了精度，但巨大的计算开销依然存在，且**未能减少 GPU 内存占用**（所有 KV Cache 仍需驻留 GPU）。

*   **核心观察（动机）**：作者发现，LLM 的注意力头在选择 Top-K 关键页面时表现出**时间稳定性差异（Temporal Stability）**：

    *   **稳定头**：在连续的解码步中，始终关注同一小批关键 token。
    *   **不稳定头**：关注的 token 集合频繁变化。
    *   更为重要的是，这种稳定性是**模型内在属性**（跨任务保持一致），只需一次离线分析即可确定。

    > ![image-20260402211549762](C:\Users\Zhaowei\AppData\Roaming\Typora\typora-user-images\image-20260402211549762.png)
    >
    > - 图片说明：
    >
    >   - 图中展示了 Llama-3.1-8B-Instruct 模型第 4 层中 8 个注意力头的“随机校正重叠率（RCO）”热力图。
    >
    >   - 横轴是解码步数的偏移量，颜色越深（越接近绿色/1.0）代表当前步关注的 Top-K 页面与后续步重叠度越高。
    >
    >   - **可以清晰地看到两类头：** 顶部的头（如 Head 0, 1）在几十步偏移内始终保持深绿色（**稳定头**，关注点极少变化）；而底部的头（如 Head 5, 6）从第一步开始就是浅色（**不稳定头**，关注的页面频繁跳跃）。这种现象是模型固有的跨任务属性，构成了后续系统设计的基石。
    >
    > - 作者采用了**定量分析与跨任务交叉验证**方法来证明这一结论：
    >
    >   1. **量化稳定性**：作者利用 **RCO（随机校正重叠率）** 指标，并结合**滑动窗口**机制(长度为W），计算出每个头在连续解码步中的平均稳定性得分（$TS_h$）。
    >
    >      ![image-20260407111007756](C:\Users\Zhaowei\Desktop\《FLEXICACHE LEVERAGING TEMPORAL STABILITY OF ATTENTION HEADS FOR EFFICIENT KV CACHE MANAGEMENT》.assets\image-20260407111007756.png)
    >
    >   2. **分类设定**：在单一任务中，系统将所有注意力头按稳定性得分排序，强制将得分最低的 **25%** 划分为“不稳定头”（例如对 Llama-3.1-8B 而言是固定的 64 个头），剩余 **75%** 为“稳定头”。
    >
    >   3. **核心交叉验证**：
    >
    >      * 为了证明这种不稳定性不挑任务，作者选取了 8 个完全不同领域的长文本任务（涵盖论文摘要、专利分析、会议记录、法律合同理解等）。
    >
    >      * 作者在每个任务上独立跑一遍上述流程，得到了 8 份不同的“64 个不稳定头名单”。
    >
    >      * **实验结果**：作者计算了这 8 份名单两两之间的重叠率（即交集大小）。如表所示，跨任务的平均重叠率高达 **0.83**，部分任务间甚至接近 0.90。
    >
    >        <img src="C:\Users\Zhaowei\Desktop\《FLEXICACHE LEVERAGING TEMPORAL STABILITY OF ATTENTION HEADS FOR EFFICIENT KV CACHE MANAGEMENT》.assets\image-20260407100518076.png" alt="image-20260407100518076" style="zoom:67%;" />
    >
    >   4. **实验结论与系统意义**：
    >
    >      *   **注意力头的不稳定性是刻在模型权重里的客观物理属性（Model-Intrinsic）**，不随输入文本或任务类型的变化而改变。
    >      *   **工程价值**：这意味着部署系统（FlexiCache）时，**只需在离线阶段跑一次性能分析（Profiling）**，找出这 25% 的不稳定头并记录下来即可。在线上提供服务时，系统无需任何动态计算开销，直接查表就能进行内存的分类分配。
    >
    >   

### 2、方法与创新

为了利用上述观察，作者提出了 **FlexiCache**，一个基于头部时间稳定性的层次化 KV Cache 管理系统。

- **优化目标：**

  - **降低注意力计算开销**

    - 采用稀疏注意力，在解码阶段，无论是稳定头还是不稳定头，系统都不会对所有的kv页面进行计算，而是**只取Top-K最重要的页面参与注意力计算**。

  - **优化 GPU 内存使用**

    - 对于稳定头：只把**Top-K页面**驻留在GPU显存中，剩下的全部offload到主机内存
    - 对于不稳定头：**关注点频繁跳跃**，因此将所有kv页面强制保留在GPU显存里

  - **最小化 I/O 传输**

    - 对于不稳定头：每一步都重新计算 Top-K（因为都在 GPU 里，算得快，且不需要走 PCIe 传输数据）。
    - 对于稳定头：周期性重排（例如每 16 步算一次）。重排后，**仅仅把新晋升的页面从 CPU 搬到 GPU**。极大地减少了 PCIe 带宽的压力。

  - **保持高质量的语言建模能力**

    - 绝不永久丢弃任何token，虽然**稳定头的非 Top-K 页面被赶出了 GPU**，但它们被安全地存放在 **CPU 内存**中。一旦未来的生成需要用到它们，系统随时可以把它们“捞”回来。这就解决了以往稀疏方法在长生成任务中精度崩溃的问题。

    ------

- **FlexiCache整体架构：**

  ![image-20260402220221227](C:\Users\Zhaowei\AppData\Roaming\Typora\typora-user-images\image-20260402220221227.png)

  > **控制器层 (Controller - 运行在 CPU 上)**
  >
  > - **FlexiCache Scheduler (调度器)：** 负责流水线管理，确保计算任务和 KV 页面从 CPU 到 GPU 的加载过程能重叠进行，不让 GPU “空转”等待数据。
  > - **FlexiCache Block Allocator (块分配器)：** 扩展了 vLLM 的 PagedAttention。它不仅管理 GPU 块，还管理主机内存块，实现“分层内存感知”的分配。
  >
  > **工作进程层 (Worker - 运行在 GPU 上)**
  >
  > - **Select Top-K (稳定性感知重排器)：** 根据头的分类，以不同的频率执行页面评分和排序。
  > - **Sparse Decode (稀疏解码内核)：** 定制的 Triton/CUDA 内核，只在选定的 Top-K 页面上执行 Attention。
  > - **KV Transfer (传输模块)：** 包含 **Page Offloader**（将新生成的页面传给 CPU）和 **Page Reloader**（将重要页面从 CPU 拉回 GPU）。



- **FlexiCache具体核心方法**：

  - **KV页评分机制**（选择topk个页进行稀疏注意力计算）

    - 采用MinMax估算法（**来自Quest论文**），对于每个 KV 页面，维护两个向量——该页面内每维度的最小和最大键值。给定当前查询向量 q，页面 p 的重要性评分如下：
      ![image-20260403112931961](C:\Users\Zhaowei\AppData\Roaming\Typora\typora-user-images\image-20260403112931961.png)

      > <img src="C:\Users\Zhaowei\AppData\Roaming\Typora\typora-user-images\image-20260403155656238.png" alt="image-20260403155656238" style="zoom:77%;" />
      >
      > MinMax 估算法图示（来自Quest论文）

    - 页面评分以**不同的频率**运行：不稳定头在每一步都被评分，而稳定头由于其时间稳定性而定期（每 16 步）评分。

      > 如果一个层仅由稳定头组成，则可以完全跳过该层的评分

    - 存储解耦 , Flexicache 将 MinMax Cache(元数据层) **永久保留在gpu**中，即使对应的 KV 页面被卸载到了 Host Memory (CPU)，它的 MinMax 边界信息也还在 GPU 里。这就使得系统能够在gpu中评估卸载页面的重要性，并且在必要时将其重新加载到 GPU 内存中。

  - **分层 KV 缓存管理**（如何在GPU显存以及主机内存之间构建并管理分层存储体系）

    - **对于不稳定头**：全量保留在GPU，因为这些头的关注点跳动很快，如果放在CPU，每一步都要搬运数据，PCIe带宽会爆掉。

    - **对于稳定头**：GPU只存tok-k个页，host存全量。因为它们很稳定，大部分时间只需要那几个 Top-K 页面，剩下的页面放在慢一点的 CPU 里即可。

    - **最小化数据传输量。**（针对稳定头）

      - 每个 KV 页面在生成后，只会被完整地传输到主机内存**一次**。之后主机内存就永远持有了页面的副本

      - 当稳定头每 16 步进行一次重排序时，系统会计算：**“有哪些页面是以前不在 Top-K 列表里，但现在新进来的？”**系统只从 Host 搬运这几个“新晋”页面（即 Delta 部分），而不是整个 Top-K 列表。

        <img src="C:\Users\Zhaowei\AppData\Roaming\Typora\typora-user-images\image-20260403155632042.png" alt="image-20260403155632042" style="zoom:50%;" />

        > 表显示了 L-Eval 中不同任务的平均提示长度（以 token 为单位）和平均生成长度。同时显示了在稳定头部的周期性重排序期间，每次请求的平均提示 KV 缓存大小。在本实验中，使用了 2048 的 top-K 大小以及具有 192 个稳定头部的 Llama-3.1-8B-Instruct 模型，实验显示稳定头部的 top-K KV 缓存总大小为 192 MB，**传输大小范围为 23% 至 35%**。

    - **计算与传输重叠。**

      <img src="C:\Users\Zhaowei\AppData\Roaming\Typora\typora-user-images\image-20260403155553009.png" alt="image-20260403155553009" style="zoom:50%;" />

      - **PO (Prompt KV Offload)：** 在 Prefill 阶段完成后，系统会启动一个低优先级的后台流，把稳定头的 KV 缓存异步搬到 Host。此时解码（D1, D2...）已经在进行了，互不干扰。完成后PO后，块管理器会释放不再位于Top-K中的页面的GPU块（显著降低了每个请求的GPU内存使用量，并支持更大的批处理大小）
      - **IO (Incremental KV Offload)：** 在解码过程中，新生成的 KV 页面一旦填满，也会异步传回 Host。
      - **R (Reload KV for promoted pages)：** 
        - 当 Request 01 需要从 Host 加载新页面（R）时，FlexiCache调度器只暂停**那个正在换数据的请求**，而 Batch 里的其他请求不用停。由于搬运的数据量很小，通常在一个解码步内就完成了，同时同一批次的不同请求在不同的时间暂停（host to device），因此在给定的一小段时间，只有一小部分请求在休息。

    - **减少碎片化开销。**

      - **问题起因：**由于Paged attention 把 KV 缓存切成了一个个小块），分散地存在显存里，如果将这些分散的页传给CPU，需要将GPU端收集这些分散的块，然后复制到一个连续缓冲区，然后再在CPU端分散他们。这样的话，CPU端的收集以及分散操作会造成很大的延迟。

      - 为了消除CPU端的KVcache数据收集以及分散操作，FlexiCache自定义了一个CUDA内核，通过UVA（统一虚拟寻址）技术直接从固定的主机KVcache读取或写入。

        > 虽然直接搬运很快，但是这个操作会占用GPU的计算单元（SM），论文采取了三个技术
        >
        > 1. **低优先级后台流** ：计算流会优先进行。
        > 2. **限制规模**：规定每次读取或写入只能用一小部分的SM
        > 3. **将大型传输分割成多个较小的块**，如不是一次搬100MB，而是拆成 100 个 1MB 的小任务，一个接一个发。由于每个小的搬运任务时间都极端（**短内核**），能让高优先级的计算内核随时“插队”进来。
        >
        > 这些策略极大地减少了搬运任务造成的系统停滞，实现了“**计算为主，搬运见缝插针**”的高效并行，确保了即使在频繁交换 KV 缓存时，推理吞吐量依然稳健。

  - **高效快表管理**

    > 既然每个注意力头在显存里存的东西都不一样了，那么该如何高效地管理这些复杂的内存地址映射表？

    - 问题来源：标注的PagedAttention会将所有的注意力头和层都共用一个简单的**数据块表（Block Table）**。而FlexiCache因为每个head都在独立决定哪些页留着、哪些页踢走，导致每个头的内存布局完全不同。映射表（此表正在CPU内存中构建和更新）从原来的 (批次, 块数) 扩展到了 (批次, 层数, 头数, 块数)。

      >  表格体积增大了几百倍，如果每一步都把这张大表从 CPU 传给 GPU，**PCIe 带宽会被占满**，变成新的性能瓶颈。

    - 解决方法：

      | 解决方法               | 实现                                                         |
      | ---------------------- | ------------------------------------------------------------ |
      | 脏页追踪               | 既然表格太大搬不动，那就只搬**变动过**的部分。FlexiCache 会追踪 CPU 端表格的哪些区域被修改了（即“脏”区域）。只有发生重排序或产生新 Token 的那部分表格数据会被传送到 GPU。通过一个专门的 CUDA 内核进行碎片化的快速传输，极大地减少了 PCIe 的负载。 |
      | 重排期间的物理块复用   | 如果每次稳定头换 Top-K 页面时，都先释放旧显存块、再申请新显存块，频繁调用 CPU 的内存分配器会产生巨大的开销。FlexiCache 实现了一个**融合CUDA 内核**，当需要把页面 A（现在不重要了）换成页面 B（新晋重要）时，内核直接把原来属于 A 的物理显存块**指派**给 B。避免了显存块的频繁申请和释放。 |
      | 通过“空块”实现布局统一 | 有些头可能存了 128 个页，有些头可能刚开始存。如果不加管理，映射表会变得参差不齐，这让 GPU 的硬件加速（向量化计算）很难受 。FlexiCache 强制所有头的映射表在逻辑上保持整齐、密集的结构。 对于那些被踢出 GPU 的页面，它们在表里的位置并不会消失，而是被指向一个特殊的“**空地址**”。这个空块永远不会被注意力内核读取，但它保证了表格数据结构的**整齐划一**。 |

      

### 3、实验设计与结果分析

* **实验设置**：

  *   **软件环境：** 基于 **vLLM** 开发，修改了 **Triton** 版 Flash-Decoding 内核（仅对每个头的 Top-K 页面执行解码注意力）。使用CUDA实现专门的KV-transfer 内核（将其编译成了pytorch扩展，与vLLM集成）

  *   **硬件配置：** 使用了顶级的 **NVIDIA H100 (94GB HBM)** GPU，宿主机配备 2 个 AMD EPYC 9554 64 核处理器和 1.1 TB DDR5 系统内存。

  *   **测试模型：** 主要针对 **Llama-3.1-8B-Instruct** 和 **Mistral-7B-Instruct-v0.2**。

  *   **指标:** 

      1. 模型准确性：FlexiCache 下的基准分数与同一模型下密集注意力机制的基准分数之比
      2. 系统性能：离线推理的**总 token 吞吐量** 和 在线服务的每输出 **token 时间** (TPOT)

  *    **对比基准（Baselines）：** 

      1. **vLLM 原生版**（全量注意力）。

      2. **LServe**

         > LServe 是概念上最相关的系统。两个框架都支持分页注意力，并减少了注意力计算的有效 token 预算以及驻留在 GPU 内存中的 KV 缓存的 token 数量。
         >
         > 关键区别在于 LServe 永久丢弃了不太重要 token 的 KV 缓存，而 FlexiCache 将它们卸载到主机内存以供将来重用。

  *   **准确率**：

      * **长上下文测试 (Table 2 - LongBench)：**

        <img src="C:\Users\Zhaowei\Desktop\《FLEXICACHE LEVERAGING TEMPORAL STABILITY OF ATTENTION HEADS FOR EFFICIENT KV CACHE MANAGEMENT》.assets\image-20260407103741191.png" alt="image-20260407103741191" style="zoom:50%;" />

        *   在 16 个不同的长文本任务中，FlexiCache得分与密集注意力得分的平均比率表明，FlexiCache在两种架构中都能有效保持模型性能。
        *   **结论：** 在长文本任务中，FlexiCache 几乎是**无损**的。

      * **长生成测试 (Table 3 - L-Eval)：**

        <img src="C:\Users\Zhaowei\Desktop\《FLEXICACHE LEVERAGING TEMPORAL STABILITY OF ATTENTION HEADS FOR EFFICIENT KV CACHE MANAGEMENT》.assets\image-20260407104135553.png" alt="image-20260407104135553" style="zoom:50%;" />

        **关键发现：** 

        *   如果没有重排序，精度会掉到 0.88-0.89。
        *   **加上周期性重排序和不稳定头管理后（FlexiCache 2048个token预算）**，精度回升到了 **0.99**。

        *   **对比 LServe：** FlexiCache 的精度表现优于 LServe（0.99 vs 0.94），证明了其“不永久丢弃 Token”策略的优越性。

* **端到端效率**：

  *   **离线吞吐量 (Figure 5)：**

      <img src="C:\Users\Zhaowei\Desktop\《FLEXICACHE LEVERAGING TEMPORAL STABILITY OF ATTENTION HEADS FOR EFFICIENT KV CACHE MANAGEMENT》.assets\image-20260407104808337.png" alt="image-20260407104808337" style="zoom:50%;" />

      *   **图 (a) 和 (b)：** 随着输出长度（Output Length）增加，vLLM 的吞吐量由于显存压力剧烈下降，而 FlexiCache 表现稳健。
      *   **图 (c)：** 在 Llama-3.1-8B 上，FlexiCache 实现了 **1.46倍** 的请求吞吐量提升。
      *   **原理：** 因为每个请求占用的显存变小了，同一时间 GPU 能塞进更多的请求（Batch Size 变大）。

  *   **在线服务性能 (Figure 6)：**

      <img src="C:\Users\Zhaowei\Desktop\《FLEXICACHE LEVERAGING TEMPORAL STABILITY OF ATTENTION HEADS FOR EFFICIENT KV CACHE MANAGEMENT》.assets\image-20260407104929561.png" alt="image-20260407104929561" style="zoom:50%;" />

      *   **图 (a) 平均 TPOT（每 Token 延迟）：** 在高请求频率下，FlexiCache 的延迟远低于 vLLM。
      *   **图 (b) 平均 TTFT（首字延迟）：** vLLM 在请求率达到 0.35 时彻底“崩了”（延迟飙升），因为显存满了，新请求只能排队。而 **FlexiCache 能撑到 0.40 以上**，依然保持极低延迟。
      *   **结论：** FlexiCache 极大地延缓了系统因为显存耗尽而进入“排队灾难”的时间。

  *   **微基准测试**：

      > 通过“解剖”系统，展示了性能提升的具体来源。

      * **解码速度 vs. Batch Size (Figure 7)：**

        <img src="C:\Users\Zhaowei\Desktop\《FLEXICACHE LEVERAGING TEMPORAL STABILITY OF ATTENTION HEADS FOR EFFICIENT KV CACHE MANAGEMENT》.assets\image-20260407105204955.png" alt="image-20260407105204955" style="zoom:50%;" />

        *   随着 Batch Size 增加，FlexiCache 的稀疏注意力内核优势越来越明显。
        *   在 Batch Size 为 40 时，解码内核比 vLLM 快了 **4倍**。

      * **稳定性感知重排的收益 (Figure 8)：**

        <img src="C:\Users\Zhaowei\Desktop\《FLEXICACHE LEVERAGING TEMPORAL STABILITY OF ATTENTION HEADS FOR EFFICIENT KV CACHE MANAGEMENT》.assets\image-20260407105312415.png" alt="image-20260407105312415" style="zoom:50%;" />

        *   对比了“全量重排（Naive）”和“稳定性感知重排（FlexiCache）”。
        *   **结论：** FlexiCache 的方案快了最高 **2.44倍**。这证明了“只对 25% 的头进行每步重排，对 75% 的头进行周期重排”这个策略极大地降低了系统开销。

      * **GPU 显存节省情况 (Figure 9)：**

        <img src="C:\Users\Zhaowei\Desktop\《FLEXICACHE LEVERAGING TEMPORAL STABILITY OF ATTENTION HEADS FOR EFFICIENT KV CACHE MANAGEMENT》.assets\image-20260407105406702.png" alt="image-20260407105406702" style="zoom:50%;" />

        *   随着序列长度（Sequence Length）增加，显存占用比例持续下降。
        *   **在 20k 长度以后，显存节省率稳定在 70% 以上。** 
        *   **结论：** 这意味着你现在可以在原本只能跑 1 个请求的显存里，跑 3 到 4 个请求。


### 4、结论与展望

*   **结论**：FlexiCache 成功揭示并利用了 LLM 注意力头在时间序列上的稳定性差异。通过的“稳定/不稳定”分类和层次化的 GPU-CPU 内存协同管理，FlexiCache 在不永久丢弃任何 Token 的前提下，大幅缩减了 GPU 内存占用并加速了注意力计算。它是一个极其适用于处理长上下文、长生成任务的高效 LLM 推理服务系统。
*   **未来展望**：
    *   **与其他推理优化技术结合**：探索将 FlexiCache 与 Prefill-Decode 分离部署、算子融合以及推测解码（Speculative Decoding）等技术结合，以获得更大的端到端收益。
    *   **扩展多级存储架构**：将现有的两级（GPU-CPU）内存层级扩展到包含 NVMe 固态硬盘或分布式内存池的多级层级，以应对超大规模上下文。
    *   **联合内存管理与集群调度**：在分布式 GPU 集群中，利用注意力头的稳定性信号来联合优化请求的批处理策略、KV 缓存放置路径和预取时机。

