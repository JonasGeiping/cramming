try:
    # Prevents torch.compile from graph breaking on einops functions. Requires einops>=0.6.1
    # https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
    from torch._dynamo import allow_in_graph
    from einops._torch_specific import allow_ops_in_compiled_graph as maybe_allow_einops_compile
except (ImportError, ModuleNotFoundError):
    def maybe_allow_einops_compile():
        pass

try:
    from torch._dynamo import skip as maybe_skip_compile
except (ImportError, ModuleNotFoundError):
    def maybe_skip_compile(cls):
        return cls