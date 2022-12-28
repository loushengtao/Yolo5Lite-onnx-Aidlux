import requests
import time

def sendmes(request_url,ids,text,headers):
    ts=str(time.time())
    type='json'
    result=requests.post(request_url+"id="+ids+"&text="+text+"&ts="+ts+"&type="+type,headers=headers)



if __name__=='main':
    id='tLO4COS'
    text="您的孩子正在用xbox玩游戏!"
    ts=str(time.time()) # 时间戳
    type='json'
    request_url="http://miaotixing.com/trigger?"

    headers={
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47'}
    sendmes(request_url,id,text,headers)