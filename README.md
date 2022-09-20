
### 南开大学实习实训
#### Todo List

| **功能名称** | **实现效果**                          | 状态 | 截止日期 |
| ------------ | ------------------------------------- | ---- | -------- |
| 车牌定位     | 在图片中定位车牌                      | done | 7.16     |
| 车牌字符分割 | 准确的分割出车牌中的字符              | done | 7.16     |
| 字符识别     | 识别出分割好的字符                    | done | 7.19     |
| Web开发      | Web项目形式的数据采集、识别、展示结果 | done | 7.19     |
| GAN图像增强  | 提高图像分辨率                        | done | 7.23     |

#### 使用说明

- 车牌识别

  - 直接点击lprweb.py运行程序，在控制台打开网址

  - 在已有网址后加上/lprweb，刷新页面

  - 点击上传图片，输出识别结果

- GAN

  - 在Samples内放一张模糊的车牌图片
  - 运行generate.py，生成图片

声明：GAN网络部分借鉴`License Plate Enhancement - From TV shows to reality`


Github地址：git@github.com:zzxvictor/License-super-resolution.git


#### 环境配置
python 3.8
tensorflow 2.9
配置环境见env.yml文件，运行`conda  create -n envname -f env.yml`（envname为环境名称）安装环境
