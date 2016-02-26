---
layout: post
title: "基于压缩感知的跟踪"
date: 2016-02-20
categories: ml
---

帮学弟看论文，关于tracking，基于压缩感知的方法。[Real-time Compressive Tracking](http://www4.comp.polyu.edu.hk/~cslzhang/CT/CT.htm)。 CSDN上有一篇挺好的[博客](http://blog.csdn.net/pbypby1987/article/details/45641081)。基本上看完这个博客也就差不多懂了。

这里大致讲讲这篇文章的思路。这里虽然用到压缩感知，但是其实只是作为一种降维方法来应用。并没有深入理论

![](http://vsooda.github.io/assets/ct/framework.png)
在a中，对于当前人脸的位置，较近位置采集正样本，远离位置采集负样本。 然后通过滤波器获取特征值。这里通过不同的卷积核获取特征值，因为特征值数目较多（10的6次方集），需要降维（压缩感知的作用，降到50维）。将特征降维后，使用朴素贝叶斯进行分类。这个降维后的每个维的值可以看作一个高斯模型来建模。

在b中，对于新帧，使用当前高斯模型获取概率，将概率相乘得到置信度，最大的则认为是目标位置。同时用这个位置更新高斯模型。

这种方法缺点很明显，并没有考虑目标尺度的变化，使用相同大小的boundingBox。肯定是有问题的。

![](http://vsooda.github.io/assets/ct/algo.png)

``以下是关键代码解读``

压缩感知权重，滤波器组位置

{% highlight c++ linenos %}
vector<vector<Rect> > features; //ct 特征相对与当前rect的偏移位置，及其长宽。每个降维特征有多个rect
vector<vector<float> > featuresWeight; //特征权重
    for (int i=0; i<_numFeature; i++)
	{
		numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));
		for (int j=0; j<numRect; j++)
		{
			rectTemp.x = cvFloor(rng.uniform(0.0, (double)(_objectBox.width - 3)));
			rectTemp.y = cvFloor(rng.uniform(0.0, (double)(_objectBox.height - 3)));
			rectTemp.width = cvCeil(rng.uniform(0.0, (double)(_objectBox.width - rectTemp.x - 2)));
			rectTemp.height = cvCeil(rng.uniform(0.0, (double)(_objectBox.height - rectTemp.y - 2)));
			features[i].push_back(rectTemp);
			weightTemp = (float)pow(-1.0, cvFloor(rng.uniform(0.0, 2.0))) / sqrt(float(numRect));
			featuresWeight[i].push_back(weightTemp);           
		}
	}
{% endhighlight %}

降维

{% highlight c++ linenos %}
	for (int i=0; i<featureNum; i++)
	{
		for (int j=0; j<sampleBoxSize; j++)
		{
			tempValue = 0.0f;
			for (size_t k=0; k<features[i].size(); k++)
			{
				xMin = _sampleBox[j].x + features[i][k].x;
				xMax = _sampleBox[j].x + features[i][k].x + features[i][k].width;
				yMin = _sampleBox[j].y + features[i][k].y;
				yMax = _sampleBox[j].y + features[i][k].y + features[i][k].height;
				tempValue += featuresWeight[i][k] *  //使用积分图快速计算卷积
					(_imageIntegral.at<float>(yMin, xMin) +
					_imageIntegral.at<float>(yMax, xMax) -
					_imageIntegral.at<float>(yMin, xMax) -
					_imageIntegral.at<float>(yMax, xMin));
			}
			_sampleFeatureValue.at<float>(i,j) = tempValue;
		}
	}
{% endhighlight %}
	
	
更新高斯模型

{% highlight c++ linenos %}
	for (int i=0; i<featureNum; i++)
	{
		meanStdDev(_sampleFeatureValue.row(i), muTemp, sigmaTemp);
		_sigma[i] = (float)sqrt( _learnRate*_sigma[i]*_sigma[i]	+ (1.0f-_learnRate)*sigmaTemp.val[0]*sigmaTemp.val[0] 
		+ _learnRate*(1.0f-_learnRate)*(_mu[i]-muTemp.val[0])*(_mu[i]-muTemp.val[0]));	// equation 6 in paper
		_mu[i] = _mu[i]*_learnRate + (1.0f-_learnRate)*muTemp.val[0];	// equation 6 in paper
	}
{% endhighlight %}
	
	
最大似然估计
	
{% highlight c++ linenos %}
	for (int j=0; j<sampleBoxNum; j++)
	{
		sumRadio = 0.0f;
		for (int i=0; i<featureNum; i++)
		{
			pPos = exp( (_sampleFeatureValue.at<float>(i,j)-_muPos[i])*(_sampleFeatureValue.at<float>(i,j)-_muPos[i]) / -(2.0f*_sigmaPos[i]*_sigmaPos[i]+1e-30) ) / (_sigmaPos[i]+1e-30);
			pNeg = exp( (_sampleFeatureValue.at<float>(i,j)-_muNeg[i])*(_sampleFeatureValue.at<float>(i,j)-_muNeg[i]) / -(2.0f*_sigmaNeg[i]*_sigmaNeg[i]+1e-30) ) / (_sigmaNeg[i]+1e-30);
			sumRadio += log(pPos+1e-30) - log(pNeg+1e-30);	// equation 4
		}
		if (_radioMax < sumRadio)
		{
			_radioMax = sumRadio;
			_radioMaxIndex = j;
		}
	}
{% endhighlight %}

