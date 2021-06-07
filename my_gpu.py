## GPU moniter V1
## Written by Wen Hongtao
## Email: hatimwen@163.com
## Some bugs occur, please move to V2
from flask import Flask, template_rendered
from flask.templating import render_template
import pandas as pd
from io import StringIO
from openssh_wrapper import SSHConnection
import os
import argparse

parser = argparse.ArgumentParser(description='GPU Status monitor')
parser.add_argument('gpu', type=str)
parser.add_argument('--delay', default=1,type=int)

app = Flask(__name__)

def get_current_user(conn):
    get_temp1 = 'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory,gpu_bus_id --format=csv,nounits,noheader'
    ret1 = conn.run(get_temp1)
    ret1 = str(ret1)
    pd1 = pd.read_csv(StringIO(ret1), header=None, names=['pid', '进程名', '已用显存MB', 'gpuid'])
    get_temp2 = 'nvidia-smi --query-gpu=index,gpu_name,temperature.gpu,utilization.gpu,memory.total --format=csv,nounits,noheader'
    ret2 = conn.run(get_temp2)
    ret2 = str(ret2)
    pd2 = pd.read_csv(StringIO(ret2), header=None, names=['显卡号', 'name', '显卡温度℃', '显卡利用率%', '显存MB'])
    return pd.concat([pd1, pd2], axis=1)

def get_pd(gpu='1', delay=1, arg=None):
    if(arg):
        args = parser.parse_args()
        gpu = args.gpu
        delay = args.delay
    conn = SSHConnection('m0'+gpu, configfile='config')
    # while True:
    df = get_current_user(conn)
    dv = df.values
    for i, item in df.iterrows():
        name = item['name']
        # 当进程数目比显卡数目多时
        if pd.isnull(name):
            # print(df[df.gpuid==item['gpuid']].index[0])
            idx_gpu = df[df.gpuid==item['gpuid']].index[0]
            dv[i][4:] = df.values[idx_gpu][4:]
            # print(df.values[i][4:-1])
    df_new = pd.DataFrame(dv, columns=df.columns._values)
    df_new.pop('gpuid')
    df_new.pop('name')
    os.system('clear')
    print(df_new)
    # time.sleep(delay)
    return df_new

@app.route('/')
def get_info():
    df1 = get_pd(gpu='1', delay=1)
    df2 = get_pd(gpu='2', delay=1)
    df3 = get_pd(gpu='3', delay=1)
    df4 = get_pd(gpu='4', delay=1)
    df7 = get_pd(gpu='7', delay=1)
    df8 = get_pd(gpu='8', delay=1)
    return render_template(
        'temp.html',
        gpu1=df1.to_html(classes='data'),
        gpu2=df2.to_html(classes='data'),
        gpu3=df3.to_html(classes='data'),
        gpu4=df4.to_html(classes='data'),
        gpu7=df7.to_html(classes='data'),
        gpu8=df8.to_html(classes='data')
    )

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0',port=1111)

