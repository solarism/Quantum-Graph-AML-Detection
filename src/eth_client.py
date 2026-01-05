import requests
import pandas as pd
import time
import os

from dotenv import load_dotenv # 需要 pip install python-dotenv
load_dotenv() # 載入 .env

class EtherscanClient:
    """
    負責與 Etherscan API 互動，獲取即時鏈上交易資料。
    對應計畫書：整合 Etherscan API 即時鏈上資料作為訓練與驗證基礎。
    """
    def __init__(self, api_key=None):
        # 建議將 API KEY 設為環境變數，或直接填入字串
        #self.api_key = api_key if api_key else "YOUR_ETHERSCAN_API_KEY_HERE"
        self.api_key = api_key if api_key else os.getenv("ETHERSCAN_API_KEY")
        self.base_url = "https://api.etherscan.io/v2/api"

    def get_internal_transactions(self, address, start_block=0, end_block=99999999):
        """
        獲取指定地址的內部交易（Internal Transactions）。
        這是偵測 DeFi 合約互動 (如 Uniswap swap) 與分層洗錢的關鍵資料。
        """
        params = {
            "chainid": "1",          # V2 必填：主網 ID
            "module": "account",
            "action": "txlistinternal",  # 關鍵：抓取內部合約呼叫
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "sort": "asc",
            "apikey": self.api_key
        }
        
        try:
            print(f"正在抓取地址 {address} 的內部交易...")
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if data["status"] == "1":
                df = pd.DataFrame(data["result"])
                # 資料型別轉換與清洗
                df['value'] = df['value'].astype(float) / 10**18  # Wei 轉 Ether
                df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s')
                print(f"✅ 成功獲取 {len(df)} 筆內部交易")
                return df
            else:
                print(f"⚠️ 無資料或 API 回傳訊息: {data['message']}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ 連線錯誤: {e}")
            return pd.DataFrame()

# 測試用區塊
if __name__ == "__main__":
    # 測試用：Uniswap V2 Router
    client = EtherscanClient(api_key="NXYQ5DV5QH21RWU39AS8S984A3W37M81R6") # 實際運行請換成您的 Key
    
    df = client.get_internal_transactions("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", start_block=17000000)
    if not df.empty:
        print(df.head())