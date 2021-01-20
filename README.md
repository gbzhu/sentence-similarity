# Sentence Similarity

For compute the semantic similarity between two sentences or short text.

# 语义相似度的方法

* 数据来源

    [STS Benchmark 数据集](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)

* 计算方法

    1. 基准方法
    2. Smooth Inverse Frequency

    
    具体实现见 [sentence_similarity.ipynb](./sentence_similarity.ipynb)，通过比较，发现使用基准方法实现效果更好

    **参考**：http://nlp.town/blog/sentence-similarity/

# App 的实现

详细实现见 [sentence_similarity](./sentence_similarity.py)

# 后期工作

* 在部署app到cloud foundry的时候，无法上传大文件，（因为采用的是google 的预训练模型，将单词转换成词向量 Word2vec，预训练文件有1.6G ），现阶段可考虑的解决方案：

    1. 在 [manifest.yaml](./manifest.yaml)文件中添加参数，能在deploy的时候支持大文件上传，详细的可见cloud foundry相关文档: [Deploying a Large App](https://docs.cloudfoundry.org/devguide/deploy-apps/large-app-deploy.html)

    2. 将Word2vec 预训练文件的下载地址 *https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz* 添加到 manifest.yaml的环境变量中，在部署app时或者部署完成之后在cloud 上去下载 ---- *试验过，没凑效*

    3. 将整个app代码及所需的文件放到docker里面，将docker传到docker hub上，cloud foundry支持通过docker 部署app，参考文档 [Deploying an App with Docker](https://docs.cloudfoundry.org/devguide/deploy-apps/push-docker.html)

* 现在的代码里面都没有加uaa（权限认证的部分），后期使用的时候是需要加上的，uaa部分可参考之前自动分类的代码 *https://github.com/gbzhu/car_fb_clsfy_based_on_bert/blob/master/client4mlf/client_uaa_ml_fd.py*

* 后期还有一个问题需要考虑，当我们在创建need的时候，需要将当前need跟以前的need进行相似度检测，当数据库中need足够多的时候，如果一条一条地拿出来跟当前need进行比较，可能会有较长时间的延迟
    - 可考虑一次将所有的旧need拿出来，只请求一次，找出与当前need最相近的一天或者几条或者没有，这需要优化当前app的处理逻辑

### 参考

* [nlp-notebooks](https://github.com/nlptown/nlp-notebooks)
