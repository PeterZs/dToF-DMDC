import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from block import NestedTensorBlock
from dual_block import NestedDualStemBlock
from cross_block import NestedCrossBlock
from masked_block import MaskedJointBlock #, NestedMaskedJointBlock
from joint_block import JointBlock

__all__ = ['NestedTensorBlock', 'NestedDualStemBlock', 'NestedCrossBlock', 'JointBlock', 'MaskedJointBlock'] # 'NestedMaskedJointBlock', 

def base_block(**kwargs):
    return NestedTensorBlock(**kwargs)

def dual_block(**kwargs):
    return NestedDualStemBlock(**kwargs)

def cross_block(**kwargs):
    return NestedCrossBlock(**kwargs)

def masked_joint_block(**kwargs):
    return MaskedJointBlock(**kwargs)

def joint_block(**kwargs):
    return JointBlock(**kwargs)
