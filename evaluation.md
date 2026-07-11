## evaluation

### 数据集评测维度

- 数据集由 75 个单轮对话和 75 个多轮对话，以及 50 个往年考试真题组成，为考察 ASR 识别准确率对系统造成的影响，我们比较了 with 和 without ASR system 对系统各个指标的影响（audio stream vs pure text），最终得到了五个不同的结果:
  - results of single-turn with / without ASR
  - results of multi-turn with / without ASR
  - results of past exams (pure text)

### metrics

![evaluator.drawio (1)](C:\Users\Administrator\Downloads\evaluator.drawio (1).png)

​															Figure1.  Metrics of Easy Exam System 

Figure1介绍了系统各个组件输入和输出相关的评测metrics，除此之外，我们也记录了处理不同样例时的系统时间花销，即latency。

###  metrics evaluation

#### chunk size and K

在当前版本下，系统的主要目标是寻找合适的chunk size 和 k（number of retrieved chunks）,  其本质上是在调节“知识的颗粒度”与“上下文的吞吐量”。Base on these，我们选择了以下指标来综合考虑这些超参数对系统的影响：

- context precision：衡量系统检索出来的所有上下文切片（Retrieved Contexts）中，真正与回答问题相关的有用信息的比例。由LLM给出结果，5分为完全正确，1分为完全错误
- context recall：衡量retrieved context中包含了多少的ground truth context，由LLM给出结果，5分为完全正确，1分为完全错误
- average latency：单个case从开始到结束的间隔时间的平均值
- groundedness，correctness and retrieval relevance

以single turn数据集为例，我们测试了六种不同的超参数组合下的分数表现并进行了排序，结果如Figure 2所示。According to Figure 2 , raising K at CS=500 steadily enhances all RAG metrics, making K=10, CS=500 the top-quality and highly recommended choice. Meanwhile, increasing Chunk Size to 1000 provides the fastest latency at K=4 but severely slows down the system and drops key metrics at K=10.  关于multi-turn 数据集的结果，请见appendix  Figure #$% 


#### Word Error Rate对performance的影响

为了比较语音输入和ASR带来的文本误差，我们也比较了在其他条件相同的情况下，使用语音作为输入和用pure text作为输入时，系统的performance差异。在我们的数据集中，single turn的WER是0.212，multi-turn的WER是0.180，and some metrics are showed below in Figure 1. for other metrics, see appendix.

总的来说，The integration of ASR introduces speech recognition noise that consistently degrades system performance compared to pure text streams. Across all configurations, switching to ASR audio input triggers a visible drop in semantic similarity, answer correctness, and answer relevance. While execution latency remains stable, textual errors from transcription directly harm retrieval precision and generation quality.


###  challenges and analysis

in this section, 我们将根据上一个section中的结果分析，并结合表现糟糕的具体样例，分析并提出改进方法

我们发现在当前系统version下，Context Precision 和 Context Recall 这两个指标的得分普遍在 2.6 到 3.5 之间, 但总的来说，随着k值的增大，context precision and context reccall 的分数有所增加。然而，对于某些特定的样例如 'case single-03-003'，随着k和chunk size的变化，其分数总是表现得非常糟糕，这表明我们的系统在某些方面存在缺陷。经过分析这个case的evaluation report. 我们发现系统根据hypothesis检索出来的context并没有包含足够的ground truth context，这揭示了hypothesis retrieval的其中一个缺陷。此外，我们还发现在某些情况下，ASR系统并不能很好的识别某些专业术语。还有，某些在所有文档中常用的词并不能很好的作为检索的目标，因为包含它们的context很多时候并不是我们想要的。

为了解决这个问题，我们考察了相关的解决方案并提出了本系统的改进方法：

1. 使用混合检索，在hypothesis retrieval的基础上增加关键词检索
2. 为了解决专业术语识别错误的问题，可以用模糊匹配（模糊查询）加以改进
3. 使用BM 25算法实现TF-IDF，以寻找真正重要的关键词









































## appendix

| Metrics                 | Single-Turn with ASR | Single-Turn Pure Text | Multi-Turn with ASR | Multi-Turn Pure Text | DL Exam No-RAG |          |
| :---------------------- | -------------------- | --------------------- | ------------------- | -------------------- | -------------- | -------- |
| BERTScore F1 Similarity |                      | Used                  | Used                | Used                 | Used           | Used     |
| Answer Correctness      |                      | Used                  | Used                | Used                 | Used           | Used     |
| Groundedness            |                      | Used                  | Used                | Used                 | Used           | Not Used |
| Answer Relevance        |                      | Used                  | Used                | Used                 | Used           | Used     |
| Execution Time Cost     |                      | Used                  | Used                | Used                 | Used           | Used     |



<img src="C:\Users\Administrator\Downloads\multi_compare.png" alt="multi_compare" style="zoom:33%;" />

<img src="C:\Users\Administrator\Downloads\overall_scores.jpeg" alt="overall_scores" style="zoom: 50%;" />

![全部指标](C:\Users\Administrator\Downloads\全部指标.png)