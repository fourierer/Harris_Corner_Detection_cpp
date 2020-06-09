# Harris_Corner_Detection_cpp
Detect Harris Corner with c++ 



1.环境配置

repo在mac的Xcode上配置了opencv环境，可以参考 https://blog.csdn.net/wo164683812/article/details/80114999 。

（1）使用homebrew直接安装opencv，不用在官网下载opencv库

```shell
brew install opencv
```

安装时间有点长，brew会自动安装很多工具。opencv安装的位置为 :/usr/local/Cellar/opencv，在路径/usr/local/Cellar/opencv/4.3.0/lib查看下是否有很有.dylib文件，有的话说明安装成功了（后续需要将这些.dylib文件导入）；

（2）在安装好了opencv之后，需要在main.cpp导入所需的头文件（e2.png）

对于header，因为我的代码加头文件是：#include "opencv2/xxx.hpp"，要想代码可以找到opencv2这个文件夹，你需要自己指定search path，以下是我的路径：

Header search Paths：/usr/local/Cellar/opencv/4.3.0/include/opencv4 ；

Library search Paths：/usr/local/Cellar/opencv/4.3.0/lib；

（3）导入.dylib文件

这一步必须完成，否则还是无法使用opencv，导入方法如图（e1.png）所示（只需要导入.dylib文件即可，我这里图省事，直接把/usr/local/Cellar/opencv/4.3.0/lib里面所有文件包括文件夹全部导入了）。



2.代码

Harris_V1是csdn别人的代码，个人认为高斯卷积部分写的不是很清楚，并且最终的效果也很差，所以在这个代码的基础上我重写了高斯卷积，响应函数以及最后对响应函数作极大值抑制几个部分的代码，见Harris_V2.

代码解读：

（代码解读部分有时间再更新）



3.可视化

见repo中的result_square.png和result_ironman.png，相应的响应函数阈值也在代码中注释说明了。



