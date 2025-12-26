# 使用輕量化 Python 鏡像
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴 (視你的資料庫套件需求而定)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製程式碼與安裝套件
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# 設定 Streamlit 運行的 Port 為 8080 (Cloud Run 預設要求)
ENV PORT=8080
EXPOSE 8080

# 執行指令
ENTRYPOINT ["streamlit", "run", "transit_accessibility_map.py", "--server.port=8080", "--server.address=0.0.0.0"]