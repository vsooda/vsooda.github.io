---
layout: post
title:  "OpenGL SuperBibe：第六章 非存储着色器"
date:   2015-01-25 
categories: ML
---

这里非存储的意思是相对于内置着色器而言的。表示在客户端编写着色器代码，再发送到服务端编译链接。
###Direct3D和OpenGL都使用了以下三种着色器[wiki]：

1. 顶点着色器处理每个顶点，将顶点的空间位置投影在屏幕上，即计算顶点的二维坐标。同时，它也负责顶点的深度缓冲（Z-Buffer）的计算。顶点着色器可以掌控顶点的位置、颜色和纹理坐标等属性，但无法生成新的顶点。顶点着色器的输出传递到流水线的下一步。如果有之后定义了几何着色器，则几何着色器会处理顶点着色器的输出数据，否则，光栅化器继续流水线任务。
2. 几何着色器可以从多边形网格中增删顶点。它能够执行对CPU来说过于繁重的生成几何结构和增加模型细节的工作。Direct3D版本10增加了支持几何着色器的API, 成为Shader Model 4.0的组成部分。OpenGL只可通过它的一个插件来使用几何着色器，但极有可能在3.1版本中该功能将会归并。几何着色器的输出连接光栅化器的输入。
3. 像素着色器(Direct3D)，常常又称为片断着色器(OpenGL)，处理来自光栅化器的数据。光栅化器已经将多边形填满并通过流水线传送至像素着色器，后者逐像素计算颜色。像素着色器常用来处理场景光照和与之相关的效果，如凸凹纹理映射和调色。名称片断着色器似乎更为准确，因为对于着色器的调用和屏幕上像素的显示并非一一对应。举个例子，对于一个像素，片断着色器可能会被调用若干次来决定它最终的颜色，那些被遮挡的物体也会被计算，直到最后的深度缓冲才将各物体前后排序。


###shader运作方式：
1. 载入顶点、片段程序进行编译链接。toonShader是程序对象。GLT_ATTRIBUTE_VERTEX，GLT_ATTRIBUTE_NORMAL是opengl定义的属性槽。将vVertex，vNormal（vp，fp使用）绑定到以上属性。（vp程序处理每个顶点，每个顶点都有对应的位置和法线。）：<br/>
toonShader = gltLoadShaderPairWithAttributes("ToonShader.vp", "ToonShader.fp", 
2. GLT_ATTRIBUTE_VERTEX, "vVertex",
   GLT_ATTRIBUTE_NORMAL, "vNormal");

3. 对于vp，fp里面的uniform变量，先获得其“句柄”
locMVP = glGetUniformLocation(toonShader, "mvpMatrix");  //uniform值是只读的

4. 运行时，传入真实值。
glUseProgram(toonShader);
glUniformMatrix4fv(locMVP, 1, GL_FALSE, transformPipeline.GetModelViewProjectionMatrix()); //设置统一值

5. 真正渲染。批次用来提交顶点。着色器进行实际的渲染    
sphereBatch.Draw();

###vp, fp 程序：

1. vp程序主要计算顶点的位置和纹理坐标（可能需要计算法线，光照等）。<br/>
in：vVertex， vNormal <br/>
out：glPosition(位置), textureCoordiate(只是知道纹理坐标。怎么知道光照强度呢？查表)

2. fp程序则进行具体的着色工作：<br/>
in：textureCoordinate <br/>
out: vFragColor

###光照计算模型
```
 vec3 vEyeNormal = normalMatrix * vNormal;
// Get vertex position in eye coordinates
vec4 vPosition4 = mvMatrix * vVertex;
vec3 vPosition3 = vPosition4.xyz / vPosition4.w;
// Get vector to light source
vec3 vLightDir = normalize(vLightPosition - vPosition3);
// Dot product gives us diffuse intensity
textureCoordinate = max(0.0, dot(vEyeNormal, vLightDir));
// Don't forget to transform the geometry!
gl_Position = mvpMatrix * vVertex;
```
###纹理采样（p201）
采样器（sampler）代表将要采样的纹理所绑定的纹理单元。在下面程序中，使用纹理贴图内置函数texture来使用插值纹理坐标，对纹理进行采样。并将颜色值分配给片段颜色

vFragColor = texture(colorTable, textureCoordinate);
<br/><br/>

```
默认对应的是OpenGL SuperBibe 第五版中文版。
```