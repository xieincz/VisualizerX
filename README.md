# VisualizerX

## Install
```shell
pip install bytecode
python setup.py install
```
## Usage
decorator the function with get_local()

### Example 1
Model Code:
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

Visualize Code:
```python
from visualizer import get_local
get_local.activate() # activate decorators (before import model)
from ... import model 

# load model and data
...
out = model(data)

cache = get_local.cache # ->  {'your_attention_function.attention_map1': [attention_map], 'your_attention_function.attention_map1': [attention_map]}
```

### Usage2
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
