from CTen import Tensor


class OpCodes:
    RANDOM = 0
    EMPTY = 1

class Dtypes:
    Float32 = 2
    Float16 = 3
    Int32 = 4
    Int16 = 5




op_create = Tensor(OpCodes.RANDOM, (2, 3), Dtypes.Float32)
op_create2 = Tensor(OpCodes.RANDOM, (2, 3), Dtypes.Float32)

op_create.repeat()
# print(op_add)