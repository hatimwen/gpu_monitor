## GPU moniter V4
## MultiProcess
## Applicable to those with more than 2 gpus
## Written by Wen Hongtao
## Email: hatimwen@163.com
from flask import Flask, template_rendered
from flask.templating import render_template
import pandas as pd
from io import StringIO
from openssh_wrapper import SSHConnection
import os
import argparse
import numpy as np
import multiprocessing

parser = argparse.ArgumentParser(description='GPU Status monitor')
parser.add_argument('gpu', type=str)
parser.add_argument('--delay', default=1,type=int)

app = Flask(__name__)

def get_current_user(conn):
    get_temp1 = 'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_bus_id --format=csv,nounits,noheader'
    ret1 = conn.run(get_temp1)
    ret1 = str(ret1)
    pd1 = pd.read_csv(StringIO(ret1), header=None, names=['pid', '进程名', '已用显存MB', 'gpuid'])
    return pd1

def get_current_gpu(conn):
    get_temp2 = 'nvidia-smi --query-gpu=index,pci.bus_id,temperature.gpu,utilization.gpu,memory.total --format=csv,nounits,noheader'
    ret2 = conn.run(get_temp2)
    ret2 = str(ret2)
    pd2 = pd.read_csv(StringIO(ret2), header=None, names=['index', 'gpubusid', '显卡温度℃', '显卡利用率%', '显存MB'])
    return pd2

def get_pd(gpu='1', delay=1, arg=None):
    if(arg):
        args = parser.parse_args()
        gpu = args.gpu
        delay = args.delay
    conn = SSHConnection('m0'+gpu, configfile='config')
    # while True:
    df_process = get_current_user(conn)
    df_gpu = get_current_gpu(conn)
    df_process_null= pd.DataFrame(None, columns=df_process.columns._values)
    df_gpu_null= pd.DataFrame(None, columns=df_gpu.columns._values)
    df_null = pd.concat([df_process_null, df_gpu], axis=1)   # processnone+gpuinfo
    df = pd.concat([df_process, df_gpu_null], axis=1)    # process+gpuinfonone
    if (df_process.values == None).all():
        df_null.pop('gpuid')
        df_null.pop('gpubusid')
        # os.system('clear')
        print(df_null)
        return df_null.to_html(classes='data')
    dv = df.values
    idx_gpu_list = []
    for i, item in df.iterrows():
        idx_gpu = df_null[df_null.gpubusid==item['gpuid']].index[0]
        idx_gpu_list.append(idx_gpu)
        dv[i][-5:] = df_null.values[idx_gpu][-5:]
    idx_gpu_list = np.array(idx_gpu_list)
    for idx_gpu in range(df_gpu._values.shape[0]):
        if (idx_gpu_list!=idx_gpu).all():
            dv = np.vstack([df_null.values[idx_gpu], dv])
    # if (idx_gpu_list==0).all():
    #     dv = np.vstack([dv, df_null.values[1]])
    # elif (idx_gpu_list==1).all():
    #     dv = np.vstack([df_null.values[0], dv])
    df_new = pd.DataFrame(dv, columns=df.columns._values)
    df_new.pop('gpuid')
    df_new.pop('gpubusid')
    # os.system('clear')
    print(df_new)
    return df_new.to_html(classes='data')


@app.route('/')
def get_info():
    pool = multiprocessing.Pool(processes=6)
    result = []
    result.append(pool.apply_async(get_pd, ('1', )))
    result.append(pool.apply_async(get_pd, ('2', )))
    result.append(pool.apply_async(get_pd, ('3', )))
    result.append(pool.apply_async(get_pd, ('4', )))
    result.append(pool.apply_async(get_pd, ('7', )))
    # result.append(pool.apply_async(get_pd, ('8', )))
    pool.close()
    pool.join()
    res = []
    for r in result:
        res.append(r.get())
    return render_template(
        'temp.html',
        gpu1=res[0],
        gpu2=res[1],
        gpu3=res[2],
        gpu4=res[3],
        gpu7=res[4],
        # gpu8=res[5]
    )

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0',port=54321)

