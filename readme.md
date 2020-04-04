##项目问题说明
1. convert()问题
- 已解决：
    - tensorflow 2.0以上版本才会自带tflite
    - 由于我们的代码使用的是keras模型，convert过程中需要使用
    from_keras_model()方法替代from_saved_model()
    - 其他部分都准备完成
- 未解决的问题
    - convert()部分无法完成，出错原因在tf_nightly包出现问题，导致import包错误。
    > 目前已经在github上找到相关issue，但是跟随issue的方法进行安装并未解决问题,
    issue反映可能是windows支持的问题，请学长运行一下看看**是否mac可以执行**
    

2. float16训练问题
- 目前运行正常，执行了lenet部分，参数为time=50，epoch=10
- 得到了训练参数，存在的问题是训练效果太好了，16和32的表现都差不多