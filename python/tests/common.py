import torch

def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (1e-4, 1e-5), torch.float16: (1e-2, 1e-3)}[x.dtype]
    return torch.allclose(x, y, atol=atol, rtol=rtol)

def do_bench(fn, flops=0, warmup=10, rep=50):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    ret = fn()
    for i in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_event.record()
    for i in range(rep):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    time_ms = start_event.elapsed_time(end_event) / rep
    return time_ms

class Benchmark:
    def __init__(self, x_names, x_vals, y_name, y_vals, y_lines, ylabel, loglog, plot_name, args):
        self.x_names = x_names
        self.x_vals = x_vals
        self.y_name = y_name
        self.y_vals = y_vals
        self.y_lines = y_lines
        self.ylabel = ylabel
        self.loglog = loglog
        self.plot_name = plot_name
        self.args = args

class Mark:
    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    def _run(self, bench):
        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.DataFrame(columns=[bench.x_names[0]] + bench.y_lines)
        for x in bench.x_vals:
            x_args = {x_name: x for x_name in bench.x_names}
            row = [self.fn(**x_args, **{bench.y_name: y}, **bench.args) for y in bench.y_vals]
            df.loc[len(df)] = [x] + row
        if bench.plot_name:
            xlabel = ' = '.join(bench.x_names)
            plot = df.plot(x=bench.x_names[0], y=bench.y_lines)
            plot.set_xlabel(xlabel)
            plot.set_ylabel(bench.ylabel)
            plot.set_title(bench.plot_name)
            plot.set_xscale('log' if bench.loglog else 'linear')
            plot.set_yscale('log' if bench.loglog else 'linear')
            plt.savefig(f'{bench.plot_name}.pdf')
        print(df)
        return df

    def run(self):
        for bench in self.benchmarks:
            self._run(bench)

def perf_report(benchmarks):
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper
