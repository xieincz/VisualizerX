# VisualizerX

## 用法
### 安装
```shell
pip install bytecode
python setup.py install
```

安装完成后，只需要用get_local装饰一下Attention的函数，forward之后就可以拿到函数内与装饰器参数同名的局部变量啦~
### Usage1
比如说，我想要函数里的`attention_map`变量：
在模型文件里，我们这么写
```python
from visualizer import get_local
@get_local('attention_map1', 'attention_map2')
def your_attention_function(*args, **kwargs):
    ...
    attention_map1 = ...
    attention_map2 = ...
    ...
    return ...
```
然后在可视化代码里，我们这么写
```python
from visualizer import get_local
get_local.activate() # 激活装饰器
from ... import model # 被装饰的模型一定要在装饰器激活之后导入！！

# load model and data
...
out = model(data)

cache = get_local.cache # ->  {'your_attention_function': [attention_map]}
```
最终就会以字典形式存在`get_local.cache`里，其中key是你的函数名,value就是一个存储attention_map的列表

### Usage2
使用Pytorch时我们往往会将模块定义成一个类，此时也是一样只要装饰类内计算出attention_map的函数即可
```python
from visualizer import get_local

class Attention(nn.Module):
    def __init__(self):
        ...
    
    @get_local('attn_map')
    def forward(self, x):
        ...
        attn_map = ...
        ...
        return ...
```
