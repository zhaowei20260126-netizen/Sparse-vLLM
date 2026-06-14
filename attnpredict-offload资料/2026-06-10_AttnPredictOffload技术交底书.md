%!TEX program = xelatex
% 编译顺序: xelatex -> bibtex -> xelatex -> xelatex
% 国家自然科学基金NSFC面上项目申请书正文模板（2023年版）version1.0
% 声明：
% 注意！！！非国家自然科学基金委官方模版！！！由个人根据官方MsWord模版制作。本模版的作者尽力使本模版和官方模版生成的PDF文件视觉效果大致一样，然而，并不保证本模版有用，也不对使用本模版造成的任何直接或间接后果负责。 不得将本模版用于商用或获取经济利益。本模版可以自由修改以满足用户自己的需要。但是如果要传播本模版，则只能传播未经修改的版本。使用本模版意味着同意上述声明。
% 强烈建议自己对照官方MsWord模板确认格式和文字是否一致，尤其是蓝字部分。
% 如有问题，可以发邮件到ryanzz@foxmail.com



\documentclass[12pt,UTF8,AutoFakeBold=2,a4paper]{ctexart} %默认小四号字。允许楷体粗体。
\usepackage[english]{babel} %支持混合语言
\usepackage[dvipsnames]{xcolor}
\usepackage{graphicx} 
\usepackage{amsmath} %更多数学符号
\usepackage{wasysym}
\usepackage[unicode]{hyperref} %提供跳转链接
\usepackage{geometry} %改改尺寸
\usepackage{gbt7714}
\usepackage{natbib}
\usepackage{ctex}
\usepackage{subfigure}  %插入多图时用子图显示的宏包
\xeCJKsetup{EmboldenFactor=3}
\IfFileExists{MSKaiti.ttf}
  {\setCJKfamilyfont{MSKai}{MSKaiti.ttf}}
  {\setCJKfamilyfont{MSKai}{FandolKai-Regular.otf}}
\IfFileExists{SIMSUN.TTC}
  {\setCJKfamilyfont{MSSong}{SIMSUN.TTC}}
  {\setCJKfamilyfont{MSSong}{FandolSong-Regular.otf}}
\newcommand{\song}{\CJKfamily{MSSong}}
\newcommand{\kai}{\CJKfamily{MSKai}}
\newcommand{\upcite}[1]{\textsuperscript{\cite{#1}}} % 上标形式引用
\renewcommand{\kaishu}{\CJKfamily{MSKai}}
\newcommand{\highlight}[1]{\textcolor[RGB]{0,112,192}{~\uline{#1}~}}
\setlength{\bibsep}{0.0pt}
%\geometry{left=3.23cm,right=3.23cm,top=2.54cm,bottom=2.54cm}
%latex的页边距比word的视觉效果要大一些，稍微调整一下
%\geometry{left=2.95cm,right=2.95cm,top=2.54cm,bottom=2.54cm}%2020
%\geometry{left=2.95cm,right=2.95cm,top=2.54cm,bottom=2.54cm}
\geometry{left=3.00cm,right=3.07cm,top=2.67cm,bottom=3.27cm}
\pagestyle{empty}
\setcounter{secnumdepth}{-2} %不让那些section和subsection自带标号，标号格式自己掌握
\definecolor{MsBlue}{RGB}{0,112,192} %Ms Word 的蓝色和latex xcolor包预定义的蓝色不一样。通过屏幕取色得到。
% Renaming floats with babel
\addto\captionsenglish{
    \renewcommand{\contentsname}{目录}
    \renewcommand{\listfigurename}{插图目录}
    \renewcommand{\listtablename}{表格}
    %\renewcommand{\refname}{\sihao 参考文献}
    \renewcommand{\refname}{\sihao \kaishu \leftline{参考文献}} %这几个字默认字号稍大，改成四号字，楷书，居左(默认居中) 根据喜好自行修改，官方模板未作要求
    \renewcommand{\abstractname}{摘要}
    \renewcommand{\indexname}{索引}
    \renewcommand{\tablename}{表}
    \renewcommand{\figurename}{图}
    } %把Figure改成‘图’，reference改成‘参考文献’。如此处理是为了避免和babel包冲突。
%定义字号
\newcommand{\chuhao}{\fontsize{42pt}{\baselineskip}\selectfont}
\newcommand{\xiaochuhao}{\fontsize{36pt}{\baselineskip}\selectfont}
\newcommand{\yihao}{\fontsize{26pt}{\baselineskip}\selectfont}
\newcommand{\erhao}{\fontsize{22pt}{\baselineskip}\selectfont}
\newcommand{\xiaoerhao}{\fontsize{18pt}{\baselineskip}\selectfont}
\newcommand{\sanhao}{\fontsize{16pt}{\baselineskip}\selectfont}
\newcommand{\sihao}{\fontsize{14pt}{\baselineskip}\selectfont}
\newcommand{\dasihao}{\fontsize{14.2pt}{\baselineskip}\selectfont}
\newcommand{\xiaosihao}{\fontsize{12pt}{\baselineskip}\selectfont}
\newcommand{\wuhao}{\fontsize{10.5pt}{\baselineskip}\selectfont}
\newcommand{\xiaowuhao}{\fontsize{9pt}{\baselineskip}\selectfont}
\newcommand{\liuhao}{\fontsize{7.875pt}{\baselineskip}\selectfont}
\newcommand{\qihao}{\fontsize{5.25pt}{\baselineskip}\selectfont}
\newcommand{\LusParagraph}[1]{\par\noindent\textbf{#1}~}


\newlength{\Qlabelwidth}
\newcounter{Qcounter} % 创建一个计数器
\renewcommand{\theQcounter}{\chinese{Qcounter}}
\newcommand{\Qsection}[1]{%
  \refstepcounter{Qcounter} % 增加计数器
  \vskip 2mm
  \settowidth{\Qlabelwidth}{{\bfseries (\theQcounter)}}
    \hangafter=1
  \setlength{\hangindent}{2.6em}
    \noindent {\dasihao \kaishu {\bfseries (\theQcounter) \textbf{#1}}
    }\par % 显示编号和内容
  % \vspace{0.15cm}
}

\newcounter{QC_Counter} % 创建一个计数器
\numberwithin{QC_Counter}{Qcounter}
\newcommand{\QComment}[1]{%
  \refstepcounter{QC_Counter} % 增加计数器


  \hangafter=1
    \setlength{\hangindent}{5.4em}
     {\dasihao \kaishu \color{MsBlue} {\kai \arabic{Qcounter}.\arabic{QC_Counter}~ } { #1}}\par % 显示编号和内容
  % \vspace{0.15cm}
}


% \newcounter{mycounter}
% \numberwithin{mycounter}{subsection}
% \newcommand{\mysubsubsectionit}[1]{%
%   \refstepcounter{mycounter} % 增加计数器
%   \vspace{0.15cm}
%   \par
% % 显示编号和内容
% \vskip 2mm
% }

%字号对照表
%二号 21pt
%四号 14
%小四 12
%五号 10.5
%设置行距 1.5倍
\renewcommand{\baselinestretch}{1.5}
\XeTeXlinebreaklocale "zh"           % 中文断行

%%%% 正文开始 %%%%
\begin{document}
\begin{center}
{\sanhao \kaishu \bfseries 报告正文}
\end{center}

% {\sihao \kaishu 参照以下提纲撰写，要求内容翔实、清晰，层次分明，标题突出。{\color{MsBlue} \bfseries 请勿删除或改动下述提纲标题及括号中的文字。}}




\Qsection{本发明要解决的技术问题是什么}
\QComment{对应现有技术的所有缺点，正面描述本发明所要解决的技术问题；本发明解决不了的，不用提供。}
\QComment{缺点以逐条罗列形式表达}
\QComment{从技术角度分析无法解决上述缺点的原因所在}

本发明名称可概括为“一种基于注意力预测的长上下文大语言模型 KV cache 分层卸载与异步预取方法及系统”。本发明面向大语言模型长上下文推理，所要解决的技术问题是在输出质量风险可控的前提下，降低长上下文解码阶段的 GPU 显存占用，并尽量避免 CPU 到 GPU 的缓存搬运阻塞模型生成过程。这里所称 KV cache，是 Transformer 注意力层在推理过程中保存的历史 Key/Value 张量。输入越长、模型层数越多，需要保存的历史 Key/Value 越多；当上下文达到数万至十余万 token 时，完整 KV cache 往往成为 GPU 显存占用和解码访存开销的主要来源。

结合现有技术，本发明针对的缺点主要包括：
\begin{enumerate}
    \item 全量注意力推理把完整历史 KV cache 长期保留在 GPU 上，显存占用随上下文长度近似线性增长，限制可支持的序列长度、批大小和并发数量；
    \item 简单 offload 虽然能把部分 KV cache 放到 CPU 内存，但如果每个解码步都临时搬运大量历史 Key/Value，CPU 到 GPU 的传输会进入生成关键路径，直接拖慢解码；
    \item 固定稀疏策略通常保留序列开头和最近窗口，难以根据当前问题动态选择远端证据 token，长文档问答、检索和摘要任务中容易出现质量波动；
    \item 注意力预测器可以预测重要历史 token，但如果在每一层、每一个解码步都运行，预测器自身的池化、前向计算和 top-k 选择会形成新的固定开销；
    \item 若注意力计算层直接处理缓存卸载、预取、回写和槽位映射，模型计算逻辑会与缓存管理逻辑强耦合，后续扩展其他稀疏方法或其他硬件层级会比较困难。
\end{enumerate}

造成上述问题的技术原因在于，历史 Key/Value 同时具有两种相互冲突的属性：一方面，它是模型访问完整上下文的必要状态，不能简单丢弃；另一方面，它又是长上下文推理中最主要的显存和访存负担。全量保留能够保证信息完整，但显存代价过高；简单卸载能够降低显存，但没有解决解码步对热点 Key/Value 的即时访问需求；固定稀疏能够减少访问数量，却缺少对远端关键信息的动态识别；逐层逐步预测能够提高选择能力，又会带来额外计算开销。因此，本发明要解决的不是单纯压缩 KV cache，而是把已有注意力预测能力转化为一套可执行的缓存分层调度方法，使完整历史仍保存在 CPU 侧，GPU 侧只驻留近期和预测可能访问的热点 Key/Value，并通过复用、预取和读视图构造减少解码等待。

\Qsection{详细介绍技术背景，并描述已有的与本发明最相近似的技术方案。}
\QComment{作为本发明基础的且帮助理解本发明公知技术内容；}
\QComment{与本发明最接近的技术方案的说明――对于方法，应说明现有方法的步骤，对于装置，应当说明结构组成及其关系。}

大语言模型通常采用 Transformer 结构。自回归推理一般分为两个阶段：预填充阶段处理输入 prompt，计算并保存各层历史 Key/Value；解码阶段每次生成一个新 token，并在每层用当前 Query 访问此前保存的 Key/Value。若不保存 KV cache，每个解码步都需要重新计算全部历史 token 的 Key 和 Value，代价很高；因此，推理系统通常把历史 Key/Value 缓存下来，在后续解码中直接读取。

最直接的已有方案是全量注意力推理。其方法步骤为：预填充阶段计算并保存全部历史 Key/Value；解码阶段将当前 Query 与全部历史 Key 计算注意力分数；对分数进行归一化后与全部历史 Value 加权求和；再将当前 token 的 Key/Value 追加到缓存中，供下一步使用。该方法的优点是实现简单、质量稳定，缺点是显存和访存量随上下文长度增加而增加。另一类已有方案是 KV cache offload，其基本做法是把部分或全部历史 Key/Value 转移到 CPU 内存，在解码时再按需要搬回 GPU。该方法能够降低 GPU 显存压力，但如果搬运范围过大或搬运时机过晚，解码步会等待数据传输完成，速度下降明显。

还有一类方法是稀疏注意力或 KV cache 压缩，例如固定保留序列开头 token、最近 token，或根据历史注意力选择一部分重要 token。这类方法减少了注意力计算需要访问的历史位置，但选择规则如果过于固定，就难以覆盖不同任务中的远距离证据。原始 AttentionPredictor 使用历史注意力分布预测未来解码步的重要 token，其流程包括收集真实注意力分数、按块池化历史注意力、用卷积神经网络预测热点位置、根据预测分数选择 top-k token，并结合开头窗口和最近窗口构造稀疏注意力掩码。该方案给出了“哪些历史 token 可能重要”的判断，但没有说明完整 KV cache 如何在 CPU 和 GPU 间分层保存、如何提前搬运、如何复用预测结果、以及注意力内核应当读取怎样的数据视图。

Sparse-vLLM 是已有稀疏推理系统，内部包含注意力层、稀疏控制器和缓存管理器等模块，能够为不同稀疏方法提供接入位置。本发明并不把 AttentionPredictor 算法或 Sparse-vLLM 系统本身作为发明点，而是在二者已有能力之上，增加一种预测驱动的 KV cache 分层卸载与异步预取机制。与本发明最接近的技术方案，可以理解为“注意力热点预测 + 稀疏推理框架”的组合；本发明补足的是该组合在长上下文解码时缺少的缓存驻留、预取、复用、回写和读视图调度方法。

\Qsection{以因果关系推理的方式推导出现有技术的缺点是什么？针对这些缺点，说明本发明的目的。}
\QComment{客观评价，现有技术的缺点是针对本发明的优点来说的，本发明不能解决的缺点不必写；基于本发明能解决的问题写出发明的目的。}
\QComment{注意：所述缺点应当是技术上的缺点，例如可以是成本高、误码率高、反应速度慢等类似问题。}
\QComment{最为关键是要从技术角度分析为什么会带来所阐述的技术效果，着重体现效果的逻辑过程。}

在长上下文推理中，输入序列长度增加会直接带来更多历史 Key/Value；模型层数增加，又会使这种缓存开销在各层累积。因此，全量注意力虽然保存了完整上下文信息，但必须让大量历史 Key/Value 长期占用 GPU 显存。当显存被 KV cache 占满后，推理系统只能降低批大小、缩短可处理上下文，或者无法承载更多并发请求。这一缺点不是由实现细节偶然造成的，而是由全量缓存和长上下文长度共同决定的。

把 KV cache 简单卸载到 CPU 后，显存压力会下降，但解码阶段仍然需要频繁访问历史 Key/Value。如果每个解码步都等到需要时才从 CPU 搬运大批 Key/Value 到 GPU，数据传输就会与模型计算串行发生。由于主机到设备的传输带宽和延迟通常弱于 GPU 显存内部访问，这种等待会直接表现为解码速度下降。若进一步采用固定稀疏规则，只保留局部窗口或少量固定位置，计算和搬运量会减少，但模型在长文档中需要访问远端证据时，关键 token 可能不在保留集合内，输出质量便会出现波动。

注意力预测器能够缓解固定规则不够灵活的问题，但预测器也需要消耗时间。它需要维护注意力历史、做池化、执行神经网络前向计算，并选择 top-k 热点位置。如果这些步骤在所有层、所有解码步上重复执行，节省下来的注意力计算和缓存搬运开销会被预测器自身抵消。基于上述因果关系，本发明的目的不是追求单一指标的极限优化，而是在已有注意力预测器和已有稀疏推理系统基础上，形成一套缓存调度方法：CPU 侧保存完整历史，GPU 侧只保留当前窗口和预测热点；预测结果在时间维度和层维度适当复用；后台提前预取下一批热点 Key/Value；注意力内核只读取缓存管理器给出的紧凑视图。这样可以在显存占用、传输等待、预测开销和质量风险之间建立可调的平衡。

\Qsection{本发明技术方案的详细阐述，应该结合流程图、原理图、电路图、时序图进行说明。（越详细越好，至少要提供2页；发明中每一功能的实现都要有相应的技术方案；所有英文缩写都应有中文注释；所有附图都应该有详细的文字描述，以别人不看附图即可明白技术方案为准；同时附图中的关键词或方框图中的注释都尽量用中文；方法专利都应该提供流程图，并提供相关的系统装置。）}

\QComment{本部分为专利申请最重要部分，需要详细提供；}
\QComment{专利必须是一个技术方案，应该阐述发明目的通过什么技术方案来实现的，不能只有原理，也不能只作功能介绍；}
\QComment{附图以方框图、黑白方式提供，不能提供彩色图例；}
\QComment{对于软件、业务方法，要提供流程图；}
\QComment{必须结合流程图、原理框图、电路图、时序图等附图进行说明，每个图都应有对应的文字描述，以他人不看附图即可明白技术方案为准。}

本发明的系统结构可由七个部分组成：大语言模型推理模块、注意力预测模块、CPU 全量缓存模块、GPU 活跃缓存模块、缓存调度模块、异步预取模块和注意力读视图构造模块。建议图 1 绘制为系统原理框图：输入 token 经大语言模型推理模块产生各层 Query、Key 和 Value；缓存调度模块同时管理 CPU 全量缓存和 GPU 活跃缓存；注意力预测模块根据历史注意力分数给出后续解码可能访问的热点 token；异步预取模块根据热点位置把相应 Key/Value 从 CPU 侧搬到 GPU 侧；读视图构造模块把 GPU 中离散的缓存槽位组织成注意力内核可读取的数据视图。图中需要标明两条数据路径：一条是模型主计算路径，另一条是后台预测与预取路径。两条路径并行存在，主路径才有机会减少等待。

在存储结构上，本发明采用 CPU full backing 和 GPU active pool 的两级结构。CPU full backing 保存完整历史 Key/Value，作为逻辑上不丢失上下文信息的后备存储；GPU active pool 只保存当前解码短期内可能访问的 Key/Value，包括起始 token、近期 token、预测热点 token 和尚未写回 CPU 的新 token。缓存调度模块为每个 token 保存逻辑位置和物理槽位的映射关系。当注意力层请求读取历史 Key/Value 时，注意力层并不直接判断某个 token 位于 CPU 还是 GPU，而是向缓存调度模块请求当前层、当前解码步的读视图。这样做的好处是，注意力计算代码保持相对稳定，卸载和预取策略可以在缓存管理层独立演进。

预填充阶段的流程建议绘制为图 2。步骤一，系统读取输入 prompt，并按推理引擎允许的块大小进入模型，逐层计算完整 Key/Value。步骤二，各层产生的 Key/Value 写入 GPU 侧临时缓存，同时缓存调度模块在 CPU 内存中为完整历史分配后备存储。步骤三，预填充结束后，系统从最后若干 query token 的真实注意力分数中提取历史注意力信息，并按 pooling block size 做池化，得到块级注意力历史。步骤四，注意力预测模块根据该历史信息计算初始热点 token 集合，为第一个解码步准备可见历史集合。该阶段不改变模型对 prompt 的完整处理，目的是建立后续解码阶段所需的缓存状态和预测状态。

解码阶段的流程建议绘制为图 3。每生成一个新 token，模型在当前层先计算该 token 的 Query、Key 和 Value，其中新 Key/Value 先写入 GPU active pool，并标记为尚未写回 CPU 的 dirty KV。随后，缓存调度模块合并三类历史位置：第一类是固定保留的 sink token，用于保存序列开头的全局信息；第二类是 recent token，用于保证局部上下文连续；第三类是注意力预测模块选出的热点 token，用于覆盖中间区域和远端证据。合并后的 token 集合构成当前解码步的可见历史集合。若集合内的 Key/Value 已经在 GPU active pool 中，系统直接构造 packed view；若部分 Key/Value 仍在 CPU full backing 中，则由预取模块或同步补取逻辑将其加载到 GPU 槽位后再构造 packed view。注意力内核只读取 packed view 中列出的 Key/Value，不需要知道其原始来源。

为了降低预测器开销，本发明设置租约复用和层间复用。一次预测得到的热点 token 集合不只用于一个解码步，而是在 reuse\_steps 指定的若干解码步内持续有效，本文称为一个租约。当租约达到设定年龄时，源层重新收集注意力分数并触发下一轮预测。对于层维度，不是每一层都运行预测器，而是通过 layer\_reuse\_stride 指定若干源层运行预测，其后的相邻层复用源层热点结果。例如步长为 4 时，可由第 0、4、8 等层产生预测结果，其间层复用相邻源层的结果。该设计的目的不是改变注意力预测器本身，而是控制预测器在推理过程中的运行频率。

异步预取过程建议绘制为图 4 的时序图。横轴为时间，纵向分为模型主计算流、预测器计算流和 CPU/GPU 数据搬运流。在第 $t$ 个解码步，源层注意力计算结束后把注意力分数提交给后台任务；后台任务更新注意力历史、运行预测器、选择 top-k 热点位置，并把这些位置对应的 Key/Value 从 CPU full backing 预取到 GPU active pool。模型主计算流继续处理后续层或后续 token。到第 $t+r$ 个解码步需要使用新租约时，如果预取已经完成，主路径直接读取 GPU active pool；如果预取尚未完成，则根据最大陈旧步数决定继续使用旧租约还是等待新租约。

dirty KV 的写回采用延迟方式。解码中新生成的 Key/Value 通常会在近期窗口内停留若干步，此时立即写回 CPU 并不能马上带来收益，反而增加设备到主机的传输和同步开销。因此，本发明允许新 Key/Value 先保留在 GPU active pool 中；当其即将离开近期窗口、或者 GPU 槽位需要被新热点占用时，再检查 CPU full backing 中是否已有副本。如果没有副本，则在驱逐前写回 CPU。这样的处理既保证完整历史仍可从 CPU 侧恢复，也避免每步都做同步写回。

本发明还设置最大陈旧步数 max\_stale\_steps。该参数用于限制旧租约在新预测结果尚未准备好时继续使用的时间。若后台预测已经完成，系统提交新租约；若后台预测尚未完成且旧租约年龄未超过最大陈旧边界，系统允许继续使用旧租约，以避免频繁等待；若旧租约超过最大陈旧边界，则主路径等待新预测和预取完成。该控制规则使预测复用有明确边界，不会无期限依赖过期热点集合。上述方法可部署于包含至少一个 CPU、至少一个 GPU、CPU 内存、GPU 显存和存储介质的推理服务器，存储介质中保存大语言模型参数、注意力预测器参数和推理程序，推理程序实现缓存管理、稀疏控制、注意力计算和异步预取等功能。

\Qsection{本发明的关键点和欲保护点是什么？}

\QComment{发明内容部分提供的是为完成一定功能的完整技术方案，在本部分是提炼出技术方案的关键创新点，列出1、2、3…，以提醒代理人注意，便于专利代理人撰写权利要求书}

本发明的关键点和欲保护点如下：

\begin{enumerate}
    \item \textbf{一种由注意力预测结果驱动的 CPU/GPU 分层 KV cache 管理方法。} 该方法把注意力预测结果用于缓存驻留决策，而不是只用于注意力掩码生成。系统在 CPU 内存中保留完整历史 Key/Value，在 GPU 显存中只维护当前解码步和近未来解码步可能访问的活跃 Key/Value。这样，完整上下文信息仍然保留在较低层级存储中，GPU 侧则避免长期保存全部历史缓存。该保护点的重点是将“预测哪些 token 重要”进一步落实为“哪些 Key/Value 驻留 GPU、哪些保留在 CPU、何时预取、何时驱逐”的缓存调度规则。

    \item \textbf{一种由起始 token、近期 token 和预测热点 token 共同构成解码可见集合的方法。} 系统不完全依赖预测器输出，而是将序列开头固定保留的 token、当前生成位置附近的近期 token、以及预测器选出的中间区域热点 token 合并，作为注意力计算实际可见的历史集合。起始 token 用于保留任务开头和提示信息，近期 token 用于保持局部上下文连续，预测热点 token 用于补充远端可能相关的证据位置。该组合规则比单一固定窗口更灵活，也比完全依赖预测器更稳妥。

    \item \textbf{一种跨解码步和跨模型层复用预测结果的刷新控制机制。} 一次预测得到的热点 token 集合作为租约，在若干解码步内持续生效；租约达到设定年龄后，源层再收集新的注意力分数并刷新预测结果。同时，系统通过层间复用步长限制预测器运行的层数，使一个源层的预测结果可供相邻若干层使用。该保护点不局限于某个固定步长数值，而在于时间维度复用和层维度复用的组合控制方式。通过该方式，可以降低预测器前向计算、池化和 top-k 选择带来的额外开销。

    \item \textbf{一种面向解码主路径的异步预取、延迟写回和紧凑读视图构造机制。} 源层注意力计算结束后，系统把注意力历史更新、热点选择、驻留规划和 CPU 到 GPU 的 Key/Value 搬运提交给后台执行，尽量与后续模型计算重叠。新生成的 Key/Value 先停留在 GPU 中，只有在需要驱逐且 CPU 后备存储尚无副本时才写回。注意力计算阶段，缓存管理器向注意力内核提供紧凑读视图，使注意力内核只处理可读取的 GPU Key/Value，不直接感知卸载和预取细节。该保护点的实质是把模型主计算路径与缓存后台任务分离。

    \item \textbf{一种带最大陈旧度边界的预测租约提交和等待机制。} 后台新预测尚未完成时，旧预测结果可以继续使用有限步数，以减少主路径等待；新预测完成后，系统提交新租约；旧租约超过最大陈旧边界时，系统等待新结果。这样，预测结果复用有明确的运行边界，旧热点集合不会被无限期使用。速度收益和质量风险被放在同一调度规则中处理，便于根据任务、模型和硬件条件调节复用强度。
\end{enumerate}

本发明的保护重点不是卷积神经网络预测器本身，也不是 Sparse-vLLM 推理系统本身，而是把已有注意力预测能力组织成可运行的 CPU/GPU 分层缓存管理、异步预取、预测复用、延迟写回和紧凑读视图构造方法。上述关键点可以分别实施，也可以组合成完整的长上下文推理缓存管理系统。

\Qsection{用推理方式推导出本发明的优点。}

\QComment{结合发明内容简单介绍，一两个自然段即可}
\QComment{可以对应3部分所要解决的技术问题或发明目的来描述。}

CPU 全量后备存储保存完整历史 Key/Value 后，GPU 不再需要长期驻留全部 KV cache，因此显存压力下降。GPU 活跃池只保存起始 token、近期 token 和预测热点 token，注意力计算读取的历史范围随之缩小，解码阶段的访存量也随之降低。与简单裁剪不同，活跃池中的 Key/Value 由固定保护窗口和预测热点共同决定，既保留了局部连续性，也给远端证据留出了进入注意力计算的机会。

预测结果复用带来的收益也比较直接。租约复用使预测器不必每个解码步运行，层间复用使预测器不必每一层运行，池化、神经网络前向和 top-k 选择的固定开销随之下降。已有实验显示，在相同实现中，不做跨步和跨层复用时解码吞吐约为 10.32 token/s；只做跨步复用后约为 37.03 token/s；同时采用跨步复用和层间复用后约为 52.33 token/s。异步预取又把热点选择和 Key/Value 搬运尽量放到后台执行，后续解码步真正需要这些 Key/Value 时，主路径等待时间可以减少。最大陈旧度控制则给预测复用加上边界，避免旧预测结果长期不更新。

从系统实现角度看，注意力层只接收缓存管理器给出的紧凑读视图，不需要直接处理 CPU 和 GPU 之间的迁移细节，模块边界更清楚，也便于在已有推理引擎中维护。实验上，在 128k 上下文、批大小为 2、输出长度为 64 的目标条件下，当前实现的 attnpredict-offload 解码吞吐约为 52.33 token/s，显存峰值约为 48.64GB；同一环境下该结果超过全量基线的解码吞吐，但预填充阶段仍较慢。质量方面，LongBench 小样本和多任务扩展实验显示不同任务存在波动，例如 qasper 上 r16 配置相对全量基线有下降。因此，本发明不把质量完全不变作为结论，而是提供一种速度、显存和质量风险可调的系统机制。

\Qsection{针对4中的技术方案，是否还有别的替代方案同样能完成发明目的？}

\QComment{如果有，请尽量写明，内容的提供可以扩大专利的保护范围，防止他人绕过本技术去实现同样的发明目的。}
\QComment{所述替代可以是部分结构、器件、方法步骤的替代，也可以是完整的技术方案。}

本发明存在以下替代实施方式：
\begin{enumerate}
    \item 注意力预测模块可以采用不同实现。当前实施例复用已有卷积神经网络预测器，也可以替换为多层感知机、Transformer 小模型、循环网络、线性模型或基于统计规则的预测器。热点选择也可以从整层共享 top-k 扩展为按注意力头分别选择、按层设置预算、按任务动态调整预算，或采用分数阈值而不是固定 top-k。只要该模块能够根据历史注意力信息输出未来可能重要的 token 或 block，就可以接入本发明的分层缓存调度机制。

    \item 存储层级和驻留池组织方式可以替换。CPU 全量后备存储可扩展为 CPU 内存、统一内存、高速固态硬盘或远端内存构成的多级存储；GPU 活跃池可按层独立维护，也可在若干层之间共享，或者按照张量并行、流水并行和多 GPU 分片方式组织。上述替代方式不改变“完整历史保存在较低层级、热点 Key/Value 驻留 GPU”的基本结构。

    \item 复用和刷新策略可以采用静态或动态方式。租约复用步数和层间复用步长可使用固定配置，也可根据注意力分布变化率、生成阶段、任务类型、序列长度、延迟目标或在线质量指标自适应调整。也可以加入 calibration\_step 机制，在部分解码步收集真实注意力分数修正历史状态。但该校准机制在当前主实施例中尚未作为默认路径使用，若采用该替代方式，应单独评估其对解码吞吐和质量的影响。

    \item 异步执行和读视图构造方式可以替换。异步预取可使用单条 CUDA stream，也可使用多条流、优先级流、线程池或事件队列；脏 Key/Value 写回可在驱逐前触发，也可按批量阈值、空闲时间、内存压力或租约切换事件触发；紧凑读视图可实现为连续拷贝、槽位索引映射、块级映射或注意力内核内直接 gather 读取。

    \item 该方法可部署于单 GPU，也可扩展到张量并行、流水并行或多机推理环境。扩展时需要将 CPU 后备存储、GPU 活跃池和热点位置映射按照并行分片同步，并保证同一逻辑 token 的 Key/Value 在不同并行分片中具有一致的租约状态和读视图描述。
\end{enumerate}

上述替代方案均不改变本发明的核心思路：用注意力预测结果驱动分层 KV cache 管理，并通过复用、预取和紧凑读视图降低长上下文解码阶段的显存和运行开销。代理人在撰写权利要求时，可将预测器类型、存储层级、租约刷新方式、预取队列实现和读视图构造方式分别作为从属保护范围展开。

\Qsection{其他有助于专利代理人理解技术的资料。}

\QComment{给代理人提供更多的信息，可以有助于代理人更好更快的完成申请文件}
\QComment{代理人并不是技术专家，交底书要代理人能看懂，尤其是背景技术和详细技术方案，一定要写的全面、清楚。}
\QComment{英文缩写有中文译文，避免使用英文单词，最好在术语解释部分给出。}
\QComment{全文对同一事物的叫法应统一，避免出现一种东西多种叫法。}
\QComment{应该阐述发明目的是通过什么技术方案来实现的，不能只有原理，也不能只做功能介绍。}

本发明当前工程实现对应的方法名为 attnpredict-offload。相关代码主要位于 attnpredict\_offload.py，该文件实现 GPU 活跃池、CPU 全量后备存储、租约状态、异步预取、脏 Key/Value 写回和紧凑读视图构造。普通注意力预测器的共享逻辑位于 attnpredict.py 和 attnpredict\_cnn.py。稀疏控制和注意力调用路径位于 sparse\_controller.py 和 attention.py。当前配置项包括 attnpredict\_reuse\_steps、attnpredict\_max\_stale\_steps、attnpredict\_layer\_reuse\_stride、num\_top\_tokens、num\_sink\_tokens 和 num\_recent\_tokens，分别控制预测结果跨步复用、旧租约最大陈旧步数、层间复用步长、总保留 token 预算、起始保留 token 数和近期保留 token 数。

已有实验资料显示，在 128k 上下文、批大小为 2、输出长度为 64、reuse\_steps 为 16、max\_stale\_steps 为 16、layer\_reuse\_stride 为 4、num\_top\_tokens 为 4096、起始 token 为 64、近期 token 为 512 的配置下，attnpredict-offload 解码吞吐约为 52.33 token/s，显存约为 48.64GB。消融实验中，不启用跨步和跨层复用时解码吞吐约为 10.32 token/s；启用跨步复用但不启用跨层复用时约为 37.03 token/s；同时启用二者时约为 52.33 token/s。该结果说明，跨步复用和跨层复用是当前实现中重要的速度来源。

质量方面，已有 LongBench 多任务质量扩展实验显示，检索任务 passage\_retrieval\_en 在 200 条样本上与全量基线持平；qasper 在 50 条样本上 r16 配置相对全量基线下降约 2.26 分；multi\_news 在 20 条样本上 r16 配置相对全量基线略高约 0.44 分。这些结果只能说明方案在已有小样本测试中没有整体失效，不能据此断言所有任务质量不变。后续若要提高质量稳定性，可继续评估更小复用步数、不同层间复用步长或校准机制。显存数字也应谨慎解释，因为不同方法可能按显存利用率预分配不同大小的缓存池，峰值显存不一定完全等同于理论缓存占用。本发明的技术价值主要在于提供 CPU 完整保存、GPU 热点驻留、按预测恢复历史 Key/Value 的系统组织方式。

\Qsection{相关技术术语的名词解释。}

\QComment{交底文件之中关键专业技术术语在行业内的标准名称及解释。}

\begin{enumerate}
    \item LLM：大语言模型，基于大规模参数进行文本理解和生成的神经网络模型。
    \item Transformer：以注意力机制为核心的神经网络结构，是当前主流大语言模型的基础结构。
    \item prefill：预填充阶段，指模型一次性处理输入 prompt 并建立初始 KV cache 的阶段。
    \item decode：解码阶段，指模型在已有 KV cache 基础上逐步生成新 token 的阶段。
    \item KV cache：键值缓存，指注意力层保存的历史 Key/Value 张量，用于避免重复计算历史 token。
    \item AttentionPredictor：注意力预测器，指根据历史注意力分数预测未来解码步可能关注哪些历史 token 的模型。
    \item Sparse-vLLM：稀疏大语言模型推理系统，指本方案所依托的已有稀疏推理框架。
    \item attention score：注意力分数，指 Query 与 Key 计算得到的相似度分数。
    \item offload：卸载，指把数据从 GPU 显存迁移到 CPU 内存或其他较低层级存储。
    \item prefetch：预取，指在真正需要数据之前提前把数据搬运到目标存储层级。
    \item hot token：热点 token，指预测或规则认为后续注意力计算可能需要访问的历史 token。
    \item sink token：起始保留 token，指序列开头固定保留的一段 token。
    \item recent token：近期 token，指靠近当前生成位置、用于保持局部上下文连续的一段 token。
    \item top-k：按预测分数从高到低选择前 k 个元素。
    \item lease：租约，指一批预测热点 token 在若干解码步内可继续使用的有效状态。
    \item reuse\_steps：复用步数，指一批预测结果计划连续使用的解码步数。
    \item max\_stale\_steps：最大陈旧步数，指旧租约在新预测未完成时允许继续使用的最大年龄。
    \item layer\_reuse\_stride：层间复用步长，指每隔多少层运行一次预测器，其余层复用源层预测结果。
    \item CPU full backing：CPU 全量后备存储，指 CPU 内存中保存的完整历史 Key/Value 副本。
    \item GPU active pool：GPU 活跃池，指 GPU 中当前可被注意力计算读取的 Key/Value 槽位池。
    \item packed view：紧凑读视图，指把若干热点 Key/Value 槽位组织成注意力内核可读取的数据视图。
    \item dirty KV：脏 Key/Value，指 GPU 上已经生成但 CPU 后备存储尚未保存副本的新 Key/Value。
    \item CUDA stream：CUDA 流，指 GPU 上用于组织异步任务执行顺序的任务队列。
    \item attention kernel：注意力内核，指在 GPU 上执行注意力计算的底层程序。
\end{enumerate}

\Qsection{参考文献}


\begin{enumerate}
    \item AttentionPredictor 原始论文或开源代码，用于说明历史注意力预测热点 token 的已有算法基础。
    \item Sparse-vLLM 项目文档和代码，用于说明本发明所依托的稀疏推理系统框架。
    \item vLLM 与 PagedAttention 相关论文和开源实现，用于说明大语言模型推理中的缓存管理背景。
    \item FlashAttention 相关论文和实现，用于说明高效注意力内核背景。
    \item StreamingLLM 相关论文，用于说明基于起始 token 和近期 token 的长上下文流式推理方法。
    \item SnapKV 相关论文，用于说明基于历史重要 token 的 KV cache 压缩方法。
    \item LongBench 数据集论文和评测代码，用于说明长上下文任务质量评测背景。
    \item 本项目实验记录：2026-05-30\_nsys最终timeline与优化结论。
    \item 本项目实验记录：2026-06-07\_优化技术方案\_实验结果版。
    \item 本项目技术报告：2026-05-30\_AttnPredictOffload最终技术报告。
\end{enumerate}


\bibliographystyle{gbt7714-numerical}
\bibliography{Cmm}

\newpage
\end{document}
