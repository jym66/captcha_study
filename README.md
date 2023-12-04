### Pytorch 验证码识别

***

### 前言
1.**本项目仅供个人记录学习使用.**

2.**利用Python的captcha生成了约1w张验证码来训练模型.**

3.**生成验证码包含大写字母、小写字母、数字.**

***
### 效果 
1. 用1w张验证码训练了15个Epoch
>Epoch [5/15], Loss: 0.1717  
Epoch [10/15], Loss: 0.1518  
Epoch [15/15], Loss: 0.1412

2.**在1000张验证码上测试正确率为约百分之89.9**

#### 使用
1.**运行captchaGen.py生成验证码**
> 会生成test_captcha和train_captcha文件夹
> 可以在setting.py里修改生成的数量

2.**运行main.py进行训练，得到一个模型 model.pth.**

3.**运行pred.py进行验证**

***
#### model.pth已上传可以测试


