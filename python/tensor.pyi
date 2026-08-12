


class OpCodes:
    EMPLTY = 0
    RANDOM = 1

class  Tensor:
    def __init__(self, *args, **kwargs):...
        
    def add(self: Tensor, other: Tensor) -> Tensor: 
        """adds an ADD op to the op graph"""
        ...
    def sub(self: Tensor, other: Tensor) -> Tensor: 
        """adds a SUB op to the op graph"""
        ...
    def mul(self: Tensor, other: Tensor) -> Tensor: 
        """adds a MUL op to the op graph"""
        ...
    def div(self: Tensor, other: Tensor) -> Tensor: 
        """adds a DIV op to the op graph"""
        ...



    def realize(self): 
        """return a tenor real tensor obect with the data"""
        return self.realize()


