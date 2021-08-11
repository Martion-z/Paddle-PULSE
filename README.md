# 复现  PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models
论文名称： PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models   
paper link：https://arxiv.org/pdf/2003.03808v3.pdf   
验收标准： 输入16*16  CelebA HQ  NIQE

# 一、代码结构
stylegan.pdparams为在FFQH数据集上预训练好的styleGan的生成器的权重   
gaussion_fit为在FFQH数据集上预训练好的styleGan的非线性映射网络  
run.py 为运行主函数   
stylegan_paddle.py 文件为styleGan的网络结构   
pulse.py 为论文PULSE提取的算法，利用预先训练好的gan不断迭代寻找最优图片   
loss.py 损失函数类
SphericalOptimizer.py 文件里为优化器类    
bicubic.py  双三次下采样类   
drive.py  驱动下载类  
niqe.py NIQE评价类


# 二、代码运行
## 1.输入
+ 输入图片需放置于input文件夹中，与[原参考代码](https://github.com/adamian98/pulse)一致，输入图片经过了预处理(aligned and downscaled),文件夹input里现有图片是对celeba数据集预处理好的图片（16*16），可直接用于测试；

(附上部分预处理好的Celeba HQ 16*16数据集百度云盘链接 链接：https://pan.baidu.com/s/1pboej7paTiWpQN1BgMIzaQ 提取码：5weo)

## 2.运行
终端执行`python3 run.py`即可运行代码，算法通过不断迭代寻找最佳输出图像，输出结果（1024x1024）存在output1024文件夹中。   
以下参数可调节:  
-input_dir default='input' 输入路径   
-output_dir default='output1024' 输出路径   
-batch_size default=1 每批次大小   
-seed 随机种子   
-eps default=2e-3 目标最小损失   
-opt_name default='adam' 优化器类别   
-steps default=100 寻找最优图片时的迭代次数   
-save_intermediate 是否存下每次迭代结果   



# 三、结果对比
## 1.视觉效果
celeba HQ数据集中随机选取了20张图片（16x16），比较其torch版本和paddle版本的输出结果（1024x1024），均进行100次迭代 
    

![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcwqt3a1fj60ps0extah02.jpg)  
![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcwr0o65tj60pd0et75z02.jpg)  
（视觉效果都挺好）

## 1.NIQE指标
从celeba HQ数据集中随机选取了20张预处理好的图片（16x16），作为torch版本和paddle版本的输入，比较其输出图片（1024*1024）的平均NIQE值(越小越好)。   
以下为结果截图:   
**torch版本：average_NIQE=2.174**
![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcwbx6ua5j60q00lltc302.jpg)
**paddle版本：average_NIQE=2.132**   
![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcx4g6u1hj60q50nkwki02.jpg)


