# VisualizerX
a modification based on https://github.com/luo3300612/Visualizer/blob/main/visualizer/visualizer.py
support to get multiple local variable from one function

## install
```shell
pip install bytecode
python setup.py install
```

## Usage
decorate the function 'with get_local'

### Example 1
```python
from visualizer import get_local
@get_local('attention_map1','attention_map2')
def your_attention_function(*args, **kwargs):
    ...
    attention_map1 = ... 
    attention_map2 = ... 
    ...
    return ...
```

Visualize
```python
from visualizer import get_local
get_local.activate() # activate decorator before import model!!
from ... import model 

# load model and data
...
out = model(data)

cache = get_local.cache # ->  {'your_attention_function.attention_map1': [attention_map], 'your_attention_function.attention_map2': [attention_map]}
```

### Example 2
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

