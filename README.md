# AgentMemEval

### 项目简介

AgentMemEval 关注的核心问题是：**一个 self-evolving agent 应该如何记忆？**

这里的“记忆”并不是简单的历史存储，而是 agent 从经历中形成自我更新能力的核心机制。
所谓 agent 在“自演化”，本质上是它的记忆结构、记忆更新方式和记忆调用方式发生变化，并进一步反映到策略与行为中。
**一个 agent 如何记录经历、组织经历、压缩经历、修订旧经验，决定了它之后如何决策、如何判断风险，以及如何形成长期行为风格。**

因此，AgentMemEval 致力于分析 self-evolving agents 中的记忆机制。
其中，德州扑克是一个合适的测试场景，因其不完全信息、高噪声等性质会放大不同 memory mechanism 之间的差异。
Agent 必须在有限观察中学习如何记录、更新和调用记忆，并将其转化为后续决策。
当然，德州扑克并不是研究的终点，而只是观察 self-evolving agents 记忆机制的一块实验切片。

### 实验设计

- **Exp1**：Memory 应该 **记录事实** 还是 **总结经验** ？<br>
    这就像是两个学习方式不同的学生：一个记性极好，把所有知识点都牢记脑海中；一个很会总结，能够把学到的内容内化成指导行为的方法论。<br>
    为此，我们设计了四种 memory 机制：

    | 方法名           | 记忆载体                                    |
    |---------------|-----------------------------------------|
    | FactAgent     | 事实库，即直接记录每轮发生的所有事实                      |
    | ExprAgent     | 经验文档，即不断把每轮发生的事情总结为经验                   |
    | FactExprSync  | Fact + Expr 两种方法并行，互不干扰                 |
    | FactExprAsync | 事实库 + 经验文档异步：事实库持续记录；异步调取相关的事实库内容迭代经验文档 |
    
    <img src="https://github.com/ckshang/AgentMemEval/blob/main/imgs/methodology1.png" width="100%">
- **Exp2**：MBTI 会怎么影响 **记忆内容** 与 **行为模式** ？<br>
    记忆很大程度上是由性格决定的。同一件事情由不同性格的人经历，也会自然而然地形成不同的记忆点入库。<br>
    为此，我们将 Exp1 中的四种 memory 机制抽象成一个统一框架，即由 LLM 控制 memory augmentation 与 memory updating 两个过程，从而使我们能够在这两个核心环节中注入 persona prompts。
    
    <img src="https://github.com/ckshang/AgentMemEval/blob/main/imgs/methodology2.png" width="50%">

### 实验 Insights
1. 在 factual memory *vs.* experiential memory 中，我们观察到明显的 **adaptation-generalization trade-off**。经验型记忆能够在训练牌桌中快速积累经验并形成有效打法，从而压制其他类型的记忆；然而，这种打法似乎是过拟合的，迁移到其他牌桌上并不适用。相比之下，事实型记忆虽然在训练牌桌中被压制，但在其他牌桌上的泛化能力更强。这和深度学习的经典问题非常相似：模型通常需要在下游任务的任务内性能与任务外泛化之间取得平衡，即 adaptation *vs.* generalization。
2. 另一个有趣的发现是，agent 的性格设定会显著影响其 memory 的形成过程，并进一步塑造其行为模式。不同性格的 agent 在记忆的筛选和更新方式上呈现出明显差异。例如，INTJ 倾向于反复自审并复修订 memory；ENFP 积累的经验最少，行为也更加冒险；等等。

**（To be continued...）**
