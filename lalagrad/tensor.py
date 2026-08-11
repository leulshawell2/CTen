from ops import Op, Ops
from typing import Tuple

class  Tensor:
    def __init__(self, *args, **kwargs):
        self.op = Op(Ops.RANDOM, args, kwargs)
        
    def __add__(self: Tensor, other: Tensor) -> Op: return self.op.add(other.op)
    def __sub__(self: Tensor, other: Tensor) -> Op: return self.op.sub(other.op)
    def __mul__(self: Tensor, other: Tensor) -> Op: return self.op.mul(other.op)
    def __div__(self: Tensor, other: Tensor) -> Op: return self.op.div(other.op)


    def repeat(self, *args: Tuple[int]): return self.op.repeat()

    def realize(self): return self.op.realize()



if __name__ == "__main__":
    a = Tensor(1, 2)
    b = Tensor(1, 2)

    c = a + b

    



