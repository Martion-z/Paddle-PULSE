# 复现  PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models
论文名称： PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models   
paper link：https://arxiv.org/pdf/2003.03808v3.pdf   
验收标准： 输入128*128  CelebA HQ 3.6

# 一、代码结构
stylegan.pdparams为在FFQH数据集上预训练好的styleGan的生成器的权重   
gaussion_fit为在FFQH数据集上预训练好的styleGan的非线性映射网络


# 二、代码运行
## 1.输入
+ 输入图片需放置于input文件夹中，并且输入的图片需要进行人脸对齐预处理，celebaHQ128文件夹内是对于celeba数据集预处理好的图片（128x128），可直接放入input用于测试；
+ 若需要测试自己的图片，可以通过align_face.py函数进行预处理：将需要预处理的图片放入init_pic里，运行align_face.py，处理后的图片会自动存入input文件夹。
## 2.运行
终端执行`python3 run.py`即可运行代码，算法通过不断迭代寻找最佳输出图像，输出结果（1024x1024）存在runs文件夹中。   
以下参数可调节:   

# 三、结果对比
celeba HQ数据集中随机选取了10张图片（128x128），比较其torch版本和paddle版本的输出结果（1024x1024）   
![image](https://github.com/Martion-z/Paddle-PULSE/illustration/compare1.png)
