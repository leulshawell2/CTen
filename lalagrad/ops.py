from enum import IntEnum
from typing import Tuple
from CTen import Ops



class OpCodes:
    RANDOM = 0


class Op: 
    def __init__(self, op: OpCodes, *args, **kwars): ...


    def add(self: Op, other: Op) -> Op: ...
    def sub(self: Op, other: Op) -> Op: ...
    def mul(self: Op, other: Op) -> Op: ...
    def div(self: Op, other: Op) -> Op: ...


    def repeat(self, *args: Tuple[int]): ...


    def realize(self): ...


    
