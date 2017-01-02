---
layout: post
title: "tinyflow源码笔记"
date: 2016-12-31
categories: code
tags: mxnet deep lua nnvm
---
* content
{:toc}

本文简单记录tinyflow阅读源码笔记。待进一步整理。文末已列出待完善部分。

```
├── src
│   ├── c_api.cc
│   ├── op_nn.cc //跟网络结构相关的op。比如conv2d等
│   ├── op_special.cc
│   ├── op_tensor.cc //矩阵运算相关op. 大量用到makenode
│   ├── op_util.h //makenode，MakeBackwardGrads。 调用nnvm
│   ├── rtc
│   │   └── op_fusion.cc
│   ├── session.cc
│   └── torch
│       ├── op_nn_torch.cc
│       ├── op_special_torch.cc
│       ├── op_tensor_torch.cc
│       └── torch_util.h
```

* 前端语言定义op,矩阵计算： op_nn.cc, op_special.cc, op_tensor.cc, op_util.h
* 后端语言: torch文件夹下。通过lua register对应torch op
* 两者的链接通过nnvm。
* 在程序运行的时候，通过session.cc拿nnvm的symbol来构造torchSession，torchExcuter。接收输入，得到输出


op_tensor_torch.cc

NNVM_REGISTER_OP(zeros)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      y[1]:fill(0)
    end
  end
)");

R 表示*Raw string literal*, 参考[这里](http://en.cppreference.com/w/cpp/language/string_literal). 表示在引号之间的字符不经过任何转义


base.h

在base.h中定义了:

using FLuaCreateNNModule = std::string;

作用等同于typedef，但是在模板中更加易用。[ref](http://stackoverflow.com/questions/10747810/what-is-the-difference-between-typedef-and-using-in-c11)

内部代码直接调用。



### session.cc

torchSession 继承自session

python 调用sessionrun调用到c api，再调用torchSession run -> torchExcute run

run接收输入，并返回输出

#### torchSession

run:session run的条件就是symbol和input

根据output节点的计算hash值。
*   1 如果hash值已经在cache_execs中，如果new_sym.output的总数目，每个output的数目，有不同，则是陈腐的(stable),如果陈腐则不计算跳转到2**这里代码可能有问题**，否则use_count++, cache_exec run
*   2. 否则exec init; set cache_exec. return torchExecute run
    * init: setupDevice, output, varstate_, SetupAuxiliaryMembers();
        * node_state, var_state在SetupAuxiliaryMembers中得到初始化

`这里的node用到nnvm的graph。这里是否stable是处于什么考虑。具体细节还需要细看`

#### torchExecutor

run:

* setup
* copy from data\_entry\_ to place holder
* op\_execs\_\[i\]\(\)
* copy from data\_entry\_ to output



setup:

* SetupShapeDType(inputs, &need_redo_infer);
    * 测试是否需要重新构造计算图,如果需要则infer shape
* if (need_redo_infer) SetupStorage();
    * SetupOpExecs 根据输入构造前向后向计算lua闭包
* if (need_redo_infer) SetupOpExecs();
* copy inputs

infer shape:

* graph_ = ApplyPasses(std::move(graph_), {"InferShape", "InferType"});
    * nnvm/ApplyPasses (pass.cc) //infer shape, infer dtype
* for (uint32_t nid : assign_var_nids_) node_states_[nid]->ResetSpace
    * ResetSpace (session.cc)

SetupOpExecs: 与后端交互的关键点。这个例子体现的是与lua torch的交互.形式如下


```cpp
std::string lua_str = "return " + lua_compute_code[inode.source->op()];
LuaRef fcompute = lua->Eval(lua_str);
op_execs_[nid] = fcompute...
```

#### 数据结构


session:

using VarStateMap = std::unordered_map<std::string, std::shared_ptr<VarState> >;

```cpp
struct VarState {
  /*! \brief The internal internal tensor */
  LuaRef tensor;
  /*! \brief The corresponding tblob */
  TBlob blob;

  /*! \return Whether the tensor is initialized already */
  inline bool initialized() const {
    return !tensor.is_nil();
  }
  // reset the space.
  inline void ResetSpace(TShape shape, int dev_mask = kCPU, int dtype = 0) {
    if (tensor.is_nil() ||
        shape != blob.shape ||
        dev_mask != blob.dev_mask ||
        dtype != blob.dtype) {
      TorchState* th = TorchState::ThreadLocalState();
      if (tensor.is_nil()) {
        tensor = th->NewTensorEmpty(dev_mask, dtype);
      }
      th->ResetStorage(
          tensor, th->NewStorage(shape.Size(), dev_mask, dtype), shape);
      this->blob = th->GetTBlob(tensor);
    }
  }
};
```

VarStateMap states_; ~~似乎没什么用~~ 在SetupAuxiliaryMembers中初始化



### lua.h

主要是c++和lua的交互

```
template<typename F>
inline void LuaState::PRun_(F f) {
  if (option_ != kLocking) {
    StackReset reset{L_, lua_gettop(L_)};
    if (option_ == kThreadLocal) {
      CHECK_EQ(ThreadLocalState(), this)
          << "Invoke lua from a different thread in ThreadLocal mode.";
    }
    f(L_);
    CHECK_EQ(reset.top, lua_gettop(L_));
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    StackReset reset{L_, lua_gettop(L_)};
    f(L_);
    CHECK_EQ(reset.top, lua_gettop(L_));
  }
}

inline void LuaRef::SetByPopStack_(LuaState* s) {
  CHECK(state_ == nullptr);
  lua_State* L = s->L_;
  if (!lua_isnil(L, -1)) {
    ref_ = lua_ref(L, LUA_REGISTRYINDEX);
    state_ = s;
  } else {
    lua_pop(L, 1);
  }
}

inline LuaRef LuaState::Eval(const char* lua_code) {
  LuaRef ret;
  //lambda表达式
  this->PRun_([this, lua_code, &ret](lua_State* L) {
      luaL_loadstring(L, lua_code);
      CHECK_EQ(lua_pcall(L, 0, 1, 0), 0)
          << "Lua call error: " << lua_tostring(L, -1) << '\n'
          << "---------\n"
          << lua_code
          << "\n----------";
      ret.SetByPopStack_(this);
    });
  return ret;
}

```

### torch_util.h
主要负责与torch的数据交互

### thread_local.h
使用单例保存数据，使得一个数据在线程内部不同地方可以重用。


### parameter

```cpp
//设置参数结构
struct ConvPoolParam : public dmlc::Parameter<ConvPoolParam> {
  TShape ksize;
  TShape strides;
  std::string padding;
  std::string data_format;
  bool no_bias;
  uint32_t num_filter;

  DMLC_DECLARE_PARAMETER(ConvPoolParam) {
    DMLC_DECLARE_FIELD(ksize).set_default(TShape{1, 1, 1, 1});
    DMLC_DECLARE_FIELD(strides).set_default(TShape{1, 1, 1, 1});
    DMLC_DECLARE_FIELD(padding).set_default("SAME");
    DMLC_DECLARE_FIELD(data_format).set_default("NCHW");
    DMLC_DECLARE_FIELD(no_bias).set_default(true);
    DMLC_DECLARE_FIELD(num_filter).set_default(0);
  }
};
DMLC_REGISTER_PARAMETER(ConvPoolParam);

NNVM_REGISTER_OP(_backward)
.describe("backward operator of NN module")
.set_num_outputs([] (const NodeAttrs& attrs) {
  const NNBackwardParam& param = dmlc::get<NNBackwardParam>(attrs.parsed);
  return param.forward_readonly_inputs - param.num_no_grad_inputs;
  })
.set_num_inputs([] (const NodeAttrs& attrs) {
  const NNBackwardParam& param = dmlc::get<NNBackwardParam>(attrs.parsed);
  uint32_t n = param.num_states + 1;
  if (param.need_inputs) n += param.forward_readonly_inputs;
  if (param.need_outputs) n += 1;
  return n;
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true);

inline std::vector<NodeEntry> MakeNNBackwardNode {
  //...
  nnvm::NodePtr p = nnvm::Node::Create();
  p->attrs.op = nnvm::Op::Get("_backward");
  p->attrs.name = n->attrs.name + "_backward";
  //...
}

NNVM_REGISTER_OP_GROUP(nn_module)
.set_attr<FGradient>("FGradient", MakeNNBackwardNode)
.set_attr<bool>("TBackwardNeedInputs", true)
.set_attr<bool>("TBackwardNeedOutputs", true);


//设置infershape方法
inline bool ConvPoolShape(const NodeAttrs& attrs,
                          std::vector<TShape> *ishape,
                          std::vector<TShape> *oshape) {
  const auto& param = dmlc::get<ConvPoolParam>(attrs.parsed);
  //...
}

//应用实例
NNVM_REGISTER_OP(conv2d)
.describe("Convolution operation")
.set_num_inputs([](const NodeAttrs& attrs){
    return (dmlc::get<ConvPoolParam>(attrs.parsed).no_bias? 2 : 3);
  })
.set_attr_parser(ParamParser<ConvPoolParam>)
.include("nn_module") //上面定义的op group
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    if (dmlc::get<ConvPoolParam>(attrs.parsed).no_bias) {
      return std::vector<std::string>{"data", "weight"};
    } else {
      return std::vector<std::string>{"data", "weight", "bias"};
    }
  })
.set_attr<FInferShape>("FInferShape", ConvPoolShape)
.set_attr<bool>("TBackwardNeedOutputs", false);

NNVM_REGISTER_OP(max_pool)
.describe("Max pooling")
.set_num_inputs(1)
.set_attr_parser(ParamParser<ConvPoolParam>) //设置参数
.include("nn_module")
.set_attr<FInferShape>("FInferShape", ConvPoolShape); //设置infershape方法

NNVM_REGISTER_OP(avg_pool)
.describe("Avg pooling")
.set_num_inputs(1)
.set_attr_parser(ParamParser<ConvPoolParam>)
.include("nn_module")
.set_attr<FInferShape>("FInferShape", ConvPoolShape);
```

在nnvm/op_attr_types.h中定义了：

```cpp
template<typename AttrType>
using FInferNodeEntryAttr = std::function<bool (const NodeAttrs& attrs,
                                                std::vector<AttrType> *in_attrs,
                                                std::vector<AttrType> *out_attrs)>;
using FInferShape = FInferNodeEntryAttr<TShape>;
using FInferType = FInferNodeEntryAttr<int>;
using TIsBackward = bool;
```

**parameter的具体细节还需要串起来**

在parameter.h中定义

```cpp
#define DMLC_DECLARE_PARAMETER(PType)                                   \
  static ::dmlc::parameter::ParamManager *__MANAGER__();                \
  inline void __DECLARE__(::dmlc::parameter::ParamManagerSingleton<PType> *manager) \

#define DMLC_DECLARE_FIELD(FieldName)  this->DECLARE(manager, #FieldName, FieldName)

#define DMLC_REGISTER_PARAMETER(PType)                                  \
  ::dmlc::parameter::ParamManager *PType::__MANAGER__() {               \
    static ::dmlc::parameter::ParamManagerSingleton<PType> inst(#PType); \
    return &inst.manager;                                               \
  }                                                                     \
  static DMLC_ATTRIBUTE_UNUSED ::dmlc::parameter::ParamManager&         \
  __make__ ## PType ## ParamManager__ =                                 \
      (*PType::__MANAGER__())                                           \

template<typename DType>
inline parameter::FieldEntry<DType>& DECLARE(
  parameter::ParamManagerSingleton<PType> *manager,
  const std::string &key, DType &ref) { // NOLINT(*)
    parameter::FieldEntry<DType> *e =
        new parameter::FieldEntry<DType>();
    e->Init(key, this->head(), ref);
    manager->manager.AddEntry(key, e);
    return *e;
}

```

在nnvm/op.h中

```
#define NNVM_REGISTER_OP(OpName)                                     \
  DMLC_STR_CONCAT(NNVM_REGISTER_VAR_DEF(OpName), __COUNTER__) =         \
      ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(#OpName)

#define NNVM_REGISTER_VAR_DEF(OpName)                                   \
  static DMLC_ATTRIBUTE_UNUSED ::nnvm::Op & __make_ ## NnvmOp ## _ ## OpName
```


### op_tensor.cc

```
NNVM_REGISTER_OP_GROUP(ElementwiseOpAttr)
.set_attr<bool>("IsElementWise", true)
.set_attr<FInferShape>("FInferShape", SameShape);

NNVM_REGISTER_OP(mul)
.add_alias("__mul_symbol__")
.describe("add two data together")
.set_num_inputs(2)
.include("ElementwiseOpAttr")
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("mul", n->attrs.name + "_grad_0",
                 {ograds[0], n->inputs[1]}),
        MakeNode("mul", n->attrs.name + "_grad_1",
                 {ograds[0], n->inputs[0]})
      };
});

```

### torch

op_nn_torch.cc

```
NNVM_REGISTER_OP(tanh)
.set_attr<FLuaCreateNNModule>(
  "FLuaCreateNNModule", R"(
  function(ishape, kwarg)
    return nn.Tanh()
  end
)");
//...
```

op_tensor_torch.cc

```
NNVM_REGISTER_OP(matmul)
.set_attr<FLuaCompute>(
  "FLuaCompute", R"(
  function(x, y, kwarg)
    return function()
      torch.mm(y[1], x[1], x[2])
    end
  end
)");
//...
```

torch部分NNVM_REGISTER_OP, 只设置FLuaCreateNNModule, FLuaCompute



### TODO

* nnvm　register具体完成什么。nnvm的工作机制是什么
* nnvm torch？
* tinyflow整个流程如何串起来？自己能写这个代码吗？
* dmlc parameter， 宏，singleton
* dmlc 其他代码。any等
* mxnet与nnvm
* lua.h对lua的封装？如何在其余场所中使用
* gpu？