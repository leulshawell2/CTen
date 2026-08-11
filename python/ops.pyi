from enum import IntEnum
from typing import Tuple



class OpCodes:
    EMPLTY = 0
    RANDOM = 1

class  Ops:
    def __init__(self, *args, **kwargs):...
        
    def add(self: Ops, other: Ops) -> Ops: 
        """adds an ADD op to the ops graph"""
        ...
    def sub(self: Ops, other: Ops) -> Ops: 
        """adds a SUB op to the ops graph"""
        ...
    def mul(self: Ops, other: Ops) -> Ops: 
        """adds a MUL op to the ops graph"""
        ...
    def div(self: Ops, other: Ops) -> Ops: 
        """adds a DIV op to the ops graph"""
        ...



    def realize(self): 
        """return a tenor real tensor obect with the data"""
        return self.op.realize()


