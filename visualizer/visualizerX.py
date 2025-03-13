from bytecode import Bytecode, Instr
import torch


class get_local(object):
    cache = {}
    is_activate = False
    detach_to_cpu = True

    def __init__(self, *varnames):
        """varname: turple"""
        self.varnames = varnames

    def __call__(self, func):
        if not type(self).is_activate:
            return func

        c = Bytecode.from_code(func.__code__)

        # store return variable
        extra_code = [Instr("STORE_FAST", "_res")]

        # store local variables
        for var_name in self.varnames:
            type(self).cache[func.__qualname__ + "." + var_name] = []  # create cache
            extra_code.extend(
                [Instr("LOAD_FAST", var_name), Instr("STORE_FAST", var_name + "_value")]
            )

        # push to TOS
        extra_code.extend([Instr("LOAD_FAST", "_res")])

        for var_name in self.varnames:
            extra_code.extend([Instr("LOAD_FAST", var_name + "_value")])

        extra_code.extend(
            [
                Instr("BUILD_TUPLE", 1 + len(self.varnames)),
                Instr("STORE_FAST", "_result_tuple"),
                Instr("LOAD_FAST", "_result_tuple"),
            ]
        )

        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        # callback
        def wrapper(*args, **kwargs):
            res, *values = func(*args, **kwargs)
            if type(self).detach_to_cpu:
                for var_idx in range(len(self.varnames)):
                    value = values[var_idx].detach().cpu()  # .numpy()
                    if value.dtype == torch.bfloat16:
                        value = value.to(torch.float32)
                    value = value.numpy()
                    type(self).cache[
                        func.__qualname__ + "." + self.varnames[var_idx]
                    ].append(value)
            else:
                for var_idx in range(len(self.varnames)):
                    value = values[var_idx]  # .detach().cpu().numpy()
                    type(self).cache[
                        func.__qualname__ + "." + self.varnames[var_idx]
                    ].append(value)
            return res

        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def set_detach_to_cpu(cls, detach_to_cpu=True):
        """set the detach_to_cpu parameter

        detach_to_cpu: bool, detach the tensor to cpu or not
        """
        cls.detach_to_cpu = detach_to_cpu

    @classmethod
    def activate(cls, detach_to_cpu=True):
        """activate the get_local decorator

        detach_to_cpu: bool, detach the tensor to cpu or not
        """
        cls.is_activate = True
        cls.detach_to_cpu = detach_to_cpu
