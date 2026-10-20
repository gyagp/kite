import argparse,json,os,sys,time,gc
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root/'local/.cache/qwen-image21/deps'),str(root/'local/.cache/embeddinggemma2/deps')]
import numpy as np, torch
from evo.image_generate import NativeDenoiser
parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);parser.add_argument('--mode',required=True);parser.add_argument('--variant',default='short-blocks');parser.add_argument('--library');args=parser.parse_args()
folders={'sky':'e6ff01e85b444f68a33af48aed5cb39b','poster':'c49d4e1d996c4e22a0a762704011ff39','edit':'1266ac01c5d24eb68f12e7d6c4ea2853'}
folder=root/'test-results/merged-workspace/images'/folders[args.case]
with np.load(folder/'fixtures/0.npz') as a:data=dict(a)
with np.load(folder/'prefix.npz') as a:prefix=dict(a)
backend='D3D12';capture=not args.mode.endswith('uncaptured');fp32=args.mode.startswith('fp32')
os.environ['EVO_WEBGPU_LIBRARY']=args.library or str(root/'local/artifacts/runtimes/ort-webgpu-c64c9fd764-bk64/onnxruntime_providers_webgpu.dll')
os.environ['EVO_IMAGE_FP32_ACCUMULATION']='1' if fp32 else '0'
output=root/'gitignore/image-optimization';output.mkdir(parents=True,exist_ok=True)
package=root/('local/artifacts/qwen-image-2.1-turbo-gguf-projections' if args.variant=='projections' else 'local/artifacts/qwen-image-2.1-turbo-gguf-floatzp')
engine=NativeDenoiser(package,data,prefix,capture,webgpu_backend=backend,graph_variant=args.variant)
sigmas=json.loads((folder/'fixtures/cases.json').read_text())['sample_sigmas']+[0.0]
record={'case':args.case,'mode':args.mode,'runs':[],'finite':True,'runtime':engine.runtime}
reference=np.load(folder/'image.npy')
for repeat in range(3):
 x=data['latents'].copy();steps=[];start=time.perf_counter()
 for index,sigma in enumerate(sigmas[:-1]):
  before=time.perf_counter();pred=engine(x,sigma,index)
  if not np.isfinite(pred).all():record['finite']=False;record['failure']={'repeat':repeat,'step':index+1};break
  x=torch.from_numpy(x+(sigmas[index+1]-sigma)*pred).bfloat16().float().numpy();steps.append(time.perf_counter()-before)
 if not record['finite']:break
 dot=float(np.dot(x.flatten(),reference.flatten())/(np.linalg.norm(x)*np.linalg.norm(reference)))
 record['runs'].append({'seconds':time.perf_counter()-start,'steps':steps,'cosine_to_current_fp32':dot,'max_abs_to_current_fp32':float(np.max(np.abs(x-reference)))})
 np.save(output/(args.case+'-'+args.mode+'-'+args.variant+'.npy'),x)
 print(args.case,args.mode,repeat,record['runs'][-1]['seconds'],dot,flush=True)
engine.close();(output/(args.case+'-'+args.mode+'-'+args.variant+'.json')).write_text(json.dumps(record,indent=2));print('RESULT',json.dumps({k:v for k,v in record.items() if k!='runtime'}),flush=True)
