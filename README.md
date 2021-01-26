# VED

## Requirement
```
pytorch==1.4
cupy
tensorboard
torchsnooper
tsnecuda
scikit-image
fvcore
opencv
imgaug==0.4.0
```

```shell
pip3 install torch==1.4.0 torchvision==0.5.01
```


Install flownet2 custome layers
```shell
bash ./script/install.sh
```

## Quick Start
The toolbox uses the shell file to start the training or inference process, and the parameters are as follows:

```shell
bash ./script/train.sh # bash ./script/inference.sh
-m MODEL: The name of the method
-d DATASET: The name of the dataset
-p PROJRECT_PATH(ABSOLUTE)
-g GPUS(e.g. 0,1)
-c CONFIG_NAME
-v VERBOSE
-f(only in inference.sh) INFERENCE MODEL
```

## Implement Models Components

**Note**
```
function 'def get_**(name):' in every package use for register name of model or hooks, loss function, ...
to make config file easy to call model, ... by name
```

### Dataset
We provide the abstract dataset class, and when users want to build the dataset, please inherit the abstract class. Meanwhile, we also provide the video and image reader tools that users can use it easily.

For example, if users want to build the dataset named Example, they should follow the steps:

1. Make a Python file named `example.py` in `./datatools/dataclass`and contains the following things:


```python
from datatools.abstract.anomaly_video_dataset import AbstractVideoAnomalyDataset
from datatools.abstract.tools import ImageLoader, VideoLoader
class Example(AbstractVideoAnomalyDataset):
    def custom_step(self):
        '''
        Step up the image loader or video loder
        '''
    def _get_frames(self):
        '''
        Step up the functions to get the frames
        '''
...
...
def get_example(cfg, flag, aug):
    t = Example()
```

2. Open the `__init__.py`  in `datatools/dataclass` and write the following things:

   ```python
   from .example import get_example
   def register_builtin_dataset():
       ...
   	DatasetCatalog.register('example', lambda cfg, flag, aug: get_example(cfg, flag, aug))
       ...
   ```

### Hooks

**Note**
```
Hooks use for Evaluation task or Analyze task during training process or inference (testing)  process
```

For example, users want to make a hook named `Example.ExampleTestHook`

1. Make a Python file named `example_hooks.py` in `lib/core/hook`and code the followings:

   ```python
   from .abstract.abstract_hook import HookBase
   HOOKS = ['ExampleTestHook']
   class ExampleTestHook(HookBase):
       def before_train(self):
           '''
           functions
           '''
       def after_train(self):
           '''
           functions
           '''
       def after_step(self):
           '''
           functions
           '''
       def before_step(self):
           '''
           functions
           '''
   
   def get_example_hooks(name):
       if name in HOOKS:
           t = eval(name)()
       else:
           raise Exception('The hook is not in amc_hooks')
       return t
   
   ```

2. Open the `__init__.py`  in `/core/hook/build` and add the following things:

   ```python
   from ..example_hooks import get_example_hooks
   
   def register_hooks():
       ...
       HookCatalog.register('Example.ExampleTestHook', lambda name:get_example_hooks(name))
       ...
   ```

### Loss Functions

For example, when users want to make new loss function name `ExampleLoss`, Open `expand_loss.py` or `basic_loss.py` in `./loss/functions/` and add the following things:

```python
...
...
class ExampleLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(ExampleLoss, self).__init__()

    def forward(self, gen, gt):
        '''
        functions
        '''
...
...

LOSSDICT ={
    ...
    'example_loss': ExampleLoss().cuda()
    ...
}
```

### Evaluation functions

Please refer to the codes

### Networks

For example, if you want to build a model named `Example`.

1. Make a Python file named `example_networks.py` in `./networks/parts`and code the followings:

   ```python
   ...
   class Example():
       '''
       the example networks
       '''
   def get_model_example():
       ...
       model_dict['example'] = Example()
       ...
       return model_dict
   ```

   

2. Open the `__init__.py`  in `./networks/build/` and add the following things:

   ```python
   from ..parts.example_networks import get_model_example
   def register_model():
       ...
       ModelCatalog.register('example', lambda cfg: get_model_example(cfg))
       ...
   ```


### Training
**Note**
```
Name of training file must be the same with the name of model in config file 
```

For example, if you want to build trainer for `Example` network.

1. Make a python file named `example.py` in `./core/` code the followings:

```python
from core.engine.default_engine import DefaultTrainer, DefaultInferencei

class Trainer(DefaultTrainer):
    NAME = ["Example.TRAIN"]

    def custom_setup(self):
        self.Example = self.model['example']
        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']


    def train(self, current_step):
    '''
    Refers to other codes
    '''

    def mini_eval(self, current_step):
    '''
    Refer to other codes
    '''


class Inference(DefaultInference):
    NAME = ["Example.INFERENCE"]

    def custom_setup(self):
        self.Example = self.model['example']
        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']

    def inference(self):
        for h in self._hooks:
            h.inference()
```
