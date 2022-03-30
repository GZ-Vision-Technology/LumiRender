1.编写模型检查工具，方便美术同学自行检查模型是否合法，提高开发效率	
2.实现各种filter
3.迪士尼材质 
4.模型自发光纹理解析	 	 	 
5.实现wavefront path tracing算法 （目前已完成初版）	 
6.完善blender exporter功能	 	 	 
7.接入openimageIO库，使CPU端能加载纹理正常渲染	 	 	 
8.合并Linux端与windows端代码	 	 	 
9.实现多光源算法	 	 	 
10.接入降噪器 （oidn跟optix进行过对比，oidn在处理glass材质时有非常明显的优势）	 	 	 
11.实现其他物理材质	 	 
12.实现纹理out of core渲染	 	 	 
13.实现几何out of core渲染	 	 	 
14.对接USD场景描述语言	 	 	 
15.实现各种halton，sobol等各种sampler	 	 	 
16.实现bdpt，mlt，pm等积分算法	 	 	 
17.编写install脚本，为用户提供安装包	 	完成	 Linux服务的第三方依赖问题将配置独立的脚本解决 
18.把预编译改为运行时编译	 	 	 
19.编写测试用例，测试distribution2D	 	 	 
20.BSSRDF非透明材质渲染算法集成		 
21.研究imgui，添加到交互渲染器交互界面. 高质量离线渲染结果添加tev的事实更新输出功能。	 	 	tev以socket与其执行程序通信，因此只需socket进程间通信方式即可.
22.支持motion blur	 	 	 
23.lstd/variant将tuple type 为值类型和指针类型的实现方式重构为模板偏特化实现方式	 	 	
偏特化实现实现不再存储_data数据成员，而采用offset存储相对对象成员的地址偏移量，当将主机内存上传到GPU内存时，无需再修订_data表示的内嵌对象的地址，直接通过Variant包覆对象的地址加地址偏移值计算内嵌对象的地址.
已经移植了一份标准库的实现, 通过了MSVC/clang/gcc-9的测试用例;
24.设计适配于std::vector的GPU内存分配器, 以此简化Buffer的管理	 	 	 类似于Buddle Allocator的显存分配管理系统，实现对大块显存分配的管理.
25.集成Google Test模块	 	 	 
26.OptiX RTcore负载均衡优化		梳理optixTrace的回调关系，利用continuation call的负载平衡方式将部分Trace流程分流到其他RTcore上.
27.自适应采样
28.物理材质取消部分纹理参数 （已完成）
29.材质纹理合并，降低访问内存次数
30.优化distribution内部实现，改成alias table（已完成）
31.处理降噪，albedo贴图，normal贴图