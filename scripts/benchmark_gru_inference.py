#!/usr/bin/env python3
"""Reproducibility benchmark, not an original manuscript measurement.

Benchmarks five independent 1-layer unidirectional GRU predictors with input_dim=128,
hidden_size=256 for batch sizes 1 and 64. Uses CUDA warmup/synchronization when
available, repeated measurements, and reports mean/std/median plus environment.
"""
import argparse, csv, json, statistics, time, platform
import torch
from torch import nn

class Predictor(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(128,256,1,batch_first=True); self.classifier=nn.Linear(256,1)
    def forward(self,x): return self.classifier(self.gru(x)[0][:,-1])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--sequence-length',type=int,default=32); p.add_argument('--repeats',type=int,default=30); p.add_argument('--warmup',type=int,default=10); p.add_argument('--out-json',default='results/cost/gru_inference_benchmark.json'); p.add_argument('--out-csv',default='results/cost/gru_inference_benchmark.csv'); a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ens=nn.ModuleList([Predictor() for _ in range(5)]).to(device).eval(); out=[]
    for bs in (1,64):
        x=torch.randn(bs,a.sequence_length,128,device=device)
        with torch.inference_mode():
            for _ in range(a.warmup):
                for m in ens: m(x)
            if device.type=='cuda': torch.cuda.synchronize()
            vals=[]
            for _ in range(a.repeats):
                t=time.perf_counter()
                for m in ens: m(x)
                if device.type=='cuda': torch.cuda.synchronize()
                vals.append((time.perf_counter()-t)*1000/bs)
        out.append({'batch_size':bs,'sequence_length':a.sequence_length,'repeats':a.repeats,'mean_ms_per_sample':statistics.mean(vals),'std_ms_per_sample':statistics.stdev(vals) if len(vals)>1 else 0,'median_ms_per_sample':statistics.median(vals)})
    env={'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU','pytorch':torch.__version__,'cuda_runtime':torch.version.cuda,'python':platform.python_version(),'architecture':'5-model ensemble; input_dim=128; hidden_size=256; layers=1; bidirectional=false'}
    with open(a.out_json,'w') as f: json.dump({'environment':env,'results':out},f,indent=2)
    with open(a.out_csv,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print(json.dumps({'environment':env,'results':out},indent=2))
if __name__=='__main__': main()
