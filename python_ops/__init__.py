from .ops import *
from .conv_ops import *
from .causal_conv import *

fc_stack = make_stack(affine)
conv_stack = make_stack(conv)
deconv_stack = make_stack(deconv)
