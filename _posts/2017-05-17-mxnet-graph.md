---
layout: post
title: "mxnet graph解析"
date: 2017-05-17
mathjax: false
categories: code
tags: mxnet graph
---
* content
{:toc}

本文是研究mxnet graph的一些代码记录。有待整理




```cpp
//graph init
// implement Executor::Bind, only call it once.
inline void Init(Symbol symbol,
                 const Context& default_ctx,
                 const std::map<std::string, Context>& ctx_map,
                 const std::vector<NDArray> &in_args,
                 const std::vector<NDArray> &arg_grad_store,
                 const std::vector<OpReqType> &grad_req_type,
                 const std::vector<NDArray> &aux_states,
                 Executor* shared_exec = nullptr) {
  enable_inplace_allocation_ = dmlc::GetEnv("MXNET_EXEC_ENABLE_INPLACE", true);
  prefer_bulk_execution_ = dmlc::GetEnv("MXNET_EXEC_PREFER_BULK_EXEC", true);
  if (shared_exec != NULL) {
    GraphExecutor* gexec = dynamic_cast<GraphExecutor*>(shared_exec);
    CHECK(gexec) << "Input executor for sharing memory must have GraphExecutor type.";
    shared_mem_ = gexec->shared_mem_;
  } else {
    shared_mem_ = std::make_shared<GraphStoragePool>();
  }

  CHECK_EQ(grad_req_type.size(), arg_grad_store.size());
  bool need_backward = false;
  for (auto req : grad_req_type) {
    if (req != kNullOp) need_backward = true;
  }
  this->InitGraph(symbol, default_ctx, ctx_map,
                  in_args, arg_grad_store, grad_req_type,
                  need_backward);
  this->InitDataEntryInfo(in_args, arg_grad_store, grad_req_type, aux_states);
  this->InitOperators();
  this->InitDataEntryMemory();
  this->InitResources();
  this->InitCachedOps();
  this->InitOpSegs();
}
```



```cpp
void GraphExecutor::Forward(bool is_train) {
  RunOps(is_train, 0, num_forward_nodes_);
}
```



```cpp
void GraphExecutor::RunOps(bool is_train, size_t topo_start, size_t topo_end) {
	Engine::Get()->Push(seg_op.opr, seg_op.ctx);
}
```





以fullconnect为例，

```cpp
enum FullyConnectedOpInputs {kData, kWeight, kBias};

Tensor<xpu, 2, DType> data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
        Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
Tensor<xpu, 2, DType> out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
    Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
out = dot(data, wmat.T());
if (!param_.no_bias) {
  Tensor<xpu, 1, DType> bias = in_data[fullc::kBias].get<xpu, 1, DType>(s);
  out += repmat(bias, data.size(0));
}
```



显然，weight, bias被作为输入处理，而不是作为op的数据。那么问题来了，这个weight，bias是什么时候构造的，怎么知道那一层对应哪个weight，bias，又怎么知道如何进行持久化



![](/assets/mxnet/list_arguments.png)



![](/assets/mxnet/infer_shape.png)



在inferAttr中,

```cpp
std::vector<AttrType> ishape, oshape;

//匿名函数
auto infer_step = [&](uint32_t nid, bool last_iter) {
    const auto& inode = idx[nid];
	const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();
    ishape.resize(num_inputs, empty_val);
    oshape.resize(num_outputs, empty_val);
    forward_known = finfer(inode.source->attrs, &ishape, &oshape);
    for (uint32_t i = 0; i < num_inputs; ++i) {
      rshape[idx.entry_id(inode.inputs[i])] = ishape[i];
    }
    for (uint32_t i = 0; i < num_outputs; ++i) {
      rshape[idx.entry_id(nid, i)] = oshape[i];
    }
}

do {
    if (i % 2 == 0) {
      //预测每个节点的shape
      for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
        infer_step(nid, false);
      }
    } else {
      // backward inference
      for (uint32_t i = idx.num_nodes(); i != 0; --i) {
        infer_step(i - 1, false);
      }
    }
    last_num_unknown = num_unknown;
    num_unknown = 0;
    for (size_t j = 0; j < idx.num_node_entries(); ++j) {
      if (fis_none(rshape[j])) {
        ++num_unknown;
      }
    }
    ++i;
  } while (num_unknown > 0 && last_num_unknown > num_unknown);

//ret是输入参数，Graph &&ret
ret.attrs[attr_name] = std::make_shared<any>(std::move(rshape));
```



```cpp
//main.cpp
auto resnet = ResNetSymbol(10);
std::map<std::string, NDArray> args_map;
std::map<std::string, NDArray> aux_map;

args_map["data"] = NDArray(Shape(batch_size, 3, 256, 256), Context::cpu());
args_map["data_label"] = NDArray(Shape(batch_size), Context::cpu());
resnet.InferArgsMap(Context::cpu(), &args_map, args_map);
auto *exec = resnet.SimpleBind(Context::cpu(), args_map);
```



```cpp
//inferArgsMap。 known_args就是data，label这种已知大小的参数
void Symbol::InferArgsMap(
    const Context &context, std::map<std::string, NDArray> *args_map,
    const std::map<std::string, NDArray> &known_args) const {
  //列出所有参数名字
  const auto arg_name_list = ListArguments();

  std::vector<std::vector<mx_uint> > in_shapes, aux_shapes, out_shapes;
  std::map<std::string, std::vector<mx_uint> > arg_shapes;

  //对于某些已知shape的参数，设置其shape大小
  for (const auto &arg_name : arg_name_list) {
    auto iter = known_args.find(arg_name);
    if (iter != known_args.end()) {
      arg_shapes[arg_name] = iter->second.GetShape();
    }
  }

  //预测未知大小的参数的shape
  InferShape(arg_shapes, &in_shapes, &aux_shapes, &out_shapes);

  for (size_t i = 0; i < in_shapes.size(); ++i) {
    const auto &shape = in_shapes[i];
    const auto &arg_name = arg_name_list[i];
    auto iter_arg = known_args.find(arg_name);
    if (iter_arg != known_args.end()) {
      (*args_map)[arg_name] = iter_arg->second;
    } else {
      //根据shape初始化参数
      (*args_map)[arg_name] = NDArray(shape, context, false);
      NDArray::SampleGaussian(0, 1, &(*args_map)[arg_name]);
    }
  }
}
```



从上面可以看出，参数是通过名字跟具体的op绑定，根据进行各种操作。





### 启动过程



```cpp
int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                     AtomicSymbolCreator **out_array) {
  mxnet::op::RegisterLegacyOpProp();
  mxnet::op::RegisterLegacyNDFunc();
  return NNListUniqueOps(out_size, out_array);
}
```



注册各种回调函数。

```cpp
void RegisterLegacyOpProp() {
  for (auto reg : dmlc::Registry<OperatorPropertyReg>::List()) {
    Op& op = ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(reg->name);
    if (op.attr_parser != nullptr) continue;
    auto creator = reg->body;
    auto attr_parser = [creator](NodeAttrs* attrs) {
      if (attrs->parsed.empty()) {
        ParsedOpProp op;
        op.ptr.reset(creator());
        op.Init(*attrs);
        attrs->parsed = std::move(op);//后面的OpPropListInputNames等操作都是拿这个parsed进行
      }
    };
    op.add_arguments(reg->arguments);
    op.describe(reg->description);
    // attribute parser
    op.set_attr_parser(attr_parser);
    op.set_num_inputs(OpPropNumInputs);
    op.set_num_outputs(OpPropNumOutputs);
    op.set_attr<nnvm::FListInputNames>("FListInputNames", OpPropListInputNames);
    op.set_attr<nnvm::FListOutputNames>("FListOutputNames", OpPropListOutputNames);
    op.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs", OpPropNumVisibleOutputs);
    op.set_attr<nnvm::FInferShape>("FInferShape", OpPropInferShape);
    op.set_attr<nnvm::FInferType>("FInferType", OpPropInferType);
    op.set_attr<nnvm::FMutateInputs>("FMutateInputs", OpPropMutateInputs);
    op.set_attr<nnvm::FInplaceOption>("FInplaceOption", OpPropInplaceOption);
    op.set_attr<FResourceRequest>("FResourceRequest", OpPropResourceRequest);
    op.set_attr<FCreateLayerOp>("FCreateLayerOp", OpPropCreateLayerOp);
    if (reg->key_var_num_args.length() != 0) {
      op.set_attr<std::string>("key_var_num_args", reg->key_var_num_args);
    }
    // register BackwardOps
    std::string back_op_name = "_backward_" + reg->name;
    Op& back_op = ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER__(back_op_name);
    op.set_attr<nnvm::FGradient>("FGradient", std::bind(
        OpPropGradient, &back_op,
        std::placeholders::_1, std::placeholders::_2));
    back_op.set_attr_parser(attr_parser);
    back_op.set_num_inputs(nnvm::kVarg);
    back_op.set_num_outputs(OpBackNumOutputs);
    back_op.set_attr<nnvm::FListOutputNames>("FListOutputNames", OpBackListOutputNames);
    back_op.set_attr<nnvm::FMutateInputs>("FMutateInputs", OpBackMutateInputs);
    back_op.set_attr<nnvm::FInplaceOption>("FInplaceOption", OpBackInplaceOption);
    back_op.set_attr<FResourceRequest>(
        "FResourceRequest", OpBackResourceRequest);
    back_op.set_attr<bool>("TIsLayerOpBackward", true);
    back_op.set_attr<bool>("TIsBackward", true);
  }
}
```


```cpp
struct NodeAttrs {
  const Op *op{nullptr};
  std::string name;
  std::vector<double> scalars;
  std::unordered_map<std::string, std::string> dict;
  any parsed; //parsed很重要
};
```



![](/assets/mxnet/parse_prop.png)


在ParsedOpProp中，

```cpp
//ParsedOpProp::init
arguments = ptr->ListArguments();
aux_states = ptr->ListAuxiliaryStates();
outputs = ptr->ListOutputs();
inputs = arguments;
inputs.insert(inputs.end(), aux_states.begin(), aux_states.end());
```



### graph

attach_op_execs_pass.cc


将op + input 变成layer的形式


```
AttachOpExecs
```
