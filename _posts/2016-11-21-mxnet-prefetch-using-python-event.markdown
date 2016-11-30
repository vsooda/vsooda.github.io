---
layout: post
title: "mxnet prefetch using python event"
date: 2016-11-21
categories: code
tags: python mxnet event io
---
* content
{:toc}

mxnet通过dataiter加载数据。c++中实现的dataiter都支持预加载，python源码中只有PrefetchingIter支持。当然也可以自己实现。

本文介绍普通的dataiter，已经支持预加载需要做的更改。同时分析预加载的实现机制，最后介绍一种多线程加载方案。



### 原始版本dataiter

```python
def __iter__(self):
    while (not self.is_finish()):
        data_value = mx.nd.empty((self.batch_size, self.n_ins))
        label_value = mx.nd.empty((self.batch_size, self.n_outs))
        batch_size = self.batch_size
        temp_train_set_x, temp_train_set_y = self.load_one_partition()
        n_train_batches = temp_train_set_x.shape[0] / batch_size
        print 'load data... %d', temp_train_set_x.shape[0]
        for index in xrange(n_train_batches):
            # print data_value.shape, temp_train_set_x.shape
            data_value[:] = temp_train_set_x[index*batch_size : (index+1)*batch_size]
            label_value[:] = temp_train_set_y[index*batch_size : (index+1)*batch_size]
            # print data_value.shape, label_value.shape
            data_all = [data_value]
            label_all = [label_value]
            data_names = ['data']
            label_names = ['label']
            yield SimpleBatch(data_names, data_all, label_names, label_all)
```

```python
class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.pad = 0
    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]
    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]
```

```python
train_dataiter_all = TTSIter(x_file_list = train_x_file_list, y_file_list = train_y_file_list, n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = True)
val_dataiter_all = TTSIter(x_file_list = valid_x_file_list, y_file_list = valid_y_file_list, n_ins = n_ins, n_outs = n_outs, batch_size = batch_size, sequential = sequential_training, shuffle = False)
model_dnn.train(train_dataiter, val_dataiter)
```

这种方案每次使用generator进行加载数据。为了避免频繁的io操作，实现了**load_one_partition**，进行批量加载。但是在批量加载的时候，gpu空等浪费大量时间。特别是在数据无法全部加载到内存中的时候。频繁倒腾数据使得效率慢3，4倍。

### PrefetchingIter预加载

在查找mxnet issue之后，发现PrefetchingIter可以用于预加载。由于mxnet的文档极具迷惑性。

```python
"""Base class for prefetching iterators. Takes one or more DataIters (
or any class with "reset" and "read" methods) and combine them with
prefetching. For example:
Parameters
----------
iters : DataIter or list of DataIter
    one or more DataIters (or any class with "reset" and "read" methods)
rename_data : None or list of dict
    i-th element is a renaming map for i-th iter, in the form of
    {'original_name' : 'new_name'}. Should have one entry for each entry
    in iter[i].provide_data
rename_label : None or list of dict
    Similar to rename_data
Examples
--------
iter = PrefetchingIter([NDArrayIter({'data': X1}), NDArrayIter({'data': X2})],
                       rename_data=[{'data': 'data1'}, {'data': 'data2'}])
"""
```

刚开始以为要实现read函数，又一直不生效。后来只得研究它的实现源码。研究后发现要实现的是**next**函数，已经提交pull request修复。看到rename_data以为需要手动对数据进行切分，发现不是。这个作用应该是有多个输入源，一起输入（**todo**：研究源码，了解data，label，rename_data，provide_data， provide_label的机制）

其实PrefetchingIter实现的是使用额外线程进行加载。加载后放到缓冲区。excutor在计算时候来缓冲区取数据。没有数据则等待，有数据则取出并通知加载线程继续加载。使用Event的进行通信。

修改后代码：

```python
def _get_batch(self):
    if self.n_train_batchs == 0 or self.batch_index == self.n_train_batchs :
        self.temp_train_set_x, self.temp_train_set_y = self.load_one_partition()
        self.n_train_batchs = self.temp_train_set_x.shape[0] / self.batch_size
        self.batch_index = 0
    prev_index = self.batch_index * self.batch_size
    back_index = (self.batch_index + 1) * self.batch_size
    self._data = {'data' : mx.nd.array(self.temp_train_set_x[prev_index:back_index])}
    self._label = {'label' : mx.nd.array(self.temp_train_set_y[prev_index:back_index])}
    self.batch_index += 1
    if self.n_train_batchs == 0:
        self._data = None
    #print self.batch_index
def iter_next(self):
    return not self.is_finish()
def next(self):
    if self.iter_next():
        self._get_batch()
        if self.n_train_batchs == 0: #the last iterator is not full
            raise StopIteration
        data_batch = mx.io.DataBatch(data=self._data.values(),
                               label=self._label.values(),
                               pad=self.getpad(), index=self.getindex())
        return data_batch
    else:
        raise StopIteration
@property
def provide_data(self):
    return [(k, v.shape) for k, v in self._data.items()]
@property
def provide_label(self):
    return [(k, v.shape) for k, v in self._label.items()]
```

**注意**: 学习provide_data， provide_label的构造。

调用:

```python
train_dataiter = mx.io.PrefetchingIter(train_dataiter_all)
val_dataiter = mx.io.PrefetchingIter(val_dataiter_all)
model_dnn.train(train_dataiter, val_dataiter)
```

这样修改后，在第一个epoch加载数据会较慢，后面epoch与全部加载内存速度差不多。gpu利用率达到90%左右

### PrefetchingIter实现

生产者

```python
self.current_batch = [None for i in range(self.n_iter)]
self.next_batch = [None for i in range(self.n_iter)]
def prefetch_func(self, i):
    """Thread entry"""
    while True:
        self.data_taken[i].wait()
        if not self.started:
            break
        try:
            self.next_batch[i] = self.iters[i].next()
        except StopIteration:
            self.next_batch[i] = None
        self.data_taken[i].clear()
        self.data_ready[i].set()
self.prefetch_threads = [threading.Thread(target=prefetch_func, args=[self, i]) \
                         for i in range(self.n_iter)]
for thread in self.prefetch_threads:
    thread.setDaemon(True)
    thread.start()
```


消费者

```python
def iter_next(self):
    for e in self.data_ready:
        e.wait()
    if self.next_batch[0] is None:
        for i in self.next_batch:
            assert i is None, "Number of entry mismatches between iterators"
        return False
    else:
        for batch in self.next_batch:
            assert batch.pad == self.next_batch[0].pad, \
                "Number of entry mismatches between iterators"
        self.current_batch = DataBatch(sum([batch.data for batch in self.next_batch], []),
                                       sum([batch.label for batch in self.next_batch], []),
                                       self.next_batch[0].pad,
                                       self.next_batch[0].index,
                                       provide_data=self.provide_data,
                                       provide_label=self.provide_label)
        for e in self.data_ready:
            e.clear()
        for e in self.data_taken:
            e.set()
```

使用Event进行通信。Event的基本操作包括：**set(), wait(), clear()**

一个事件由一个标识符表示, 这个标示符可以通过set()方法设为True,通过clear()方法重新设为False, wait()方法则使线程一直处于阻塞状态,直到标示符变为True。

### 多线程加载

上面代码使用的**PrefetchingIter**进行预加载。但是加载线程只有一个。还是有点慢。其实可以把这种加载方式写到加载数据处并开启多线程。具体参考[这篇文章](http://shuokay.com/2016/08/16/mxnet-io/)。由于目前速度已经满足需求，所以暂不考虑这个方案。
