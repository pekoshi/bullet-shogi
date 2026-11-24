import os,sys,time
print('probe_start', flush=True)
src=r'E:\shogi\p2shogi\data\washiun_dl\novel_ext3.psv'
out=r'E:\shogi\p2shogi\dataset_profile15_rebalance_v2_2500m\profile15_rebalance_v2_2500m_exact.psv'
print('src_exists',os.path.exists(src),'src_size',os.path.getsize(src) if os.path.exists(src) else -1, flush=True)
print('out_exists',os.path.exists(out),'out_size',os.path.getsize(out) if os.path.exists(out) else -1, flush=True)
with open(src,'rb',buffering=64*1024*1024) as f:
    b=f.read(40)
print('read_ok',len(b), flush=True)
