from CTen import *
from enum import IntEnum, auto


class OpCodes(IntEnum):
    RANDOM = auto()




op_create = Ops()
op_create2 = Ops()

op_add = op_create.add(op_create)