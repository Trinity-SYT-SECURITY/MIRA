# MIRA 即時視覺化使用指南

## ✅ 視覺化伺服器已啟動

您看到的訊息表示視覺化伺服器已成功啟動：
```
🌐 Live dashboard: http://localhost:5000
Browser opened automatically
```

## 📊 如何查看視覺化

### 方法 1: 自動開啟的瀏覽器
- 伺服器啟動時應該已自動開啟瀏覽器
- 查看是否有新的瀏覽器視窗/分頁開啟

### 方法 2: 手動開啟
如果瀏覽器沒有自動開啟，請手動訪問：
```
http://127.0.0.1:5000
```

## 🎯 您應該看到什麼

### 初始畫面
```
┌───────────────────────────────────────────────────────────────┐
│ 🧠 MIRA Transformer Visualization    [Before/After] [Status] │
├───────────────────────────────────────────────────────────────┤
│ [L0][L1][L2]...     [H1][H2][H3]...                           │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Input → Embedding → Q/K/V → Attention → MLP → Output         │
│   📝       🔢         🔀        👁️        ⚡      🎯           │
│  tokens   E[0..n]   Q|K|V   [matrix]   bars   probs          │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│ Step: 0  Loss: --  Status: Waiting  Suffix: "--"              │
└───────────────────────────────────────────────────────────────┘
```

### 執行過程中
當 main.py 進入 Phase 5 (Gradient Attacks) 時，您會看到：

1. **Input 區塊**: 顯示輸入的 tokens
2. **Embedding 區塊**: 顯示 E[0], E[1], E[2]... 向量
3. **Q/K/V 區塊**: 三列彩色條形圖
   - Query (藍色)
   - Key (紅色)  
   - Value (綠色)
4. **Attention 區塊**: 熱圖矩陣顯示注意力權重
5. **MLP 區塊**: 神經元激活條形圖
6. **Output 區塊**: 輸出機率分布

底部會即時更新：
- **Step**: 攻擊步驟 (0-30)
- **Loss**: 損失值 (逐漸下降)
- **Status**: Running/Complete
- **Suffix**: 對抗性後綴

## 🔧 如果沒看到視覺化

### 檢查 1: 瀏覽器是否開啟
```bash
# 手動開啟瀏覽器訪問
open http://127.0.0.1:5000
```

### 檢查 2: 端口是否被佔用
如果 5000 端口被佔用，修改 main.py 中的端口：
```python
server = LiveVisualizationServer(port=5001)  # 改成 5001
```

### 檢查 3: 等待 Phase 5
- Phase 1-4 是初始化階段，視覺化會顯示 "Waiting..."
- 只有到 Phase 5 (Gradient Attacks) 才會看到即時更新

## 📝 模型選擇跳過的問題

如果您想要選擇特定模型，可以：

### 方法 1: 使用 .env 文件
創建/編輯 `.env` 文件：
```bash
MODEL_NAME=EleutherAI/pythia-160m
```

### 方法 2: 修改 main.py
在 main.py 中直接設定：
```python
model_name = "EleutherAI/pythia-160m"  # 直接指定模型
```

## ⏱️ 執行時間預估

- **Pythia 70M**: 約 5-10 分鐘完成全部流程
- **Phase 5** 是視覺化最精彩的部分，會執行 30 步攻擊優化

## 🎬 下一步

讓 main.py 繼續執行，當進入 Phase 5 時：
1. 查看瀏覽器視覺化頁面
2. 觀察 Transformer 內部處理變化
3. 注意 Attention 矩陣的模式
4. 觀察損失值如何下降
