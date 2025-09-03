# pack/pack_data.py
from pathlib import Path
import json
import numpy as np
import torch
from types import SimpleNamespace as SN

def load_data(args):
    """
    從你轉好的 baseline 目錄讀取資料，建立與 GRAPE UCI loader 相容的 data 物件。
    需要四個檔案：X_norm.npy / y.npy / mask.npy / split_idx.json
    - row 節點 [0..N-1]、feature 節點 [N..N+d-1]
    - 邊為可見特徵（mask==1），並建立雙向邊；邊屬性 shape = (E,1)
    - train/test 的邊以 row split 切；MDI 的 labels 僅取 forward 半邊的值
    """
    root = Path(getattr(args, 'root'))
    X = np.load(root / 'X_norm.npy')              # (N, d)
    y = np.load(root / 'y.npy')                   # (N,)
    M = np.load(root / 'mask.npy').astype(np.uint8)  # (N, d) 1=可見
    with open(root / 'split_idx.json', 'r', encoding='utf-8') as f:
        split = json.load(f)

    N, d = X.shape
    # --- forward 邊（row -> feat+N） ---
    pos = np.argwhere(M == 1)
    obs_rows = pos[:, 0]
    obs_feats = pos[:, 1]
    src_fwd = torch.from_numpy(obs_rows).long()
    dst_fwd = torch.from_numpy(obs_feats + N).long()
    edge_index_fwd = torch.stack([src_fwd, dst_fwd], dim=0)  # (2, m)
    edge_attr_fwd = torch.from_numpy(X[M == 1].astype(np.float32)).view(-1, 1)  # (m,1)

    # --- reverse 邊（feat+N -> row），屬性同值 ---
    edge_index_rev = torch.stack([edge_index_fwd[1], edge_index_fwd[0]], dim=0)  # (2, m)
    edge_attr_rev  = edge_attr_fwd.clone()                                        # (m,1)

    # --- 全部邊（雙向拼接）---
    edge_index_all = torch.cat([edge_index_fwd, edge_index_rev], dim=1)           # (2, 2m)
    edge_attr_all  = torch.cat([edge_attr_fwd, edge_attr_rev], dim=0)             # (2m,1)

    # --- 依 row split 切訓練/測試（valid 可無）---
    tr_rows = np.asarray(split['train'], dtype=np.int64)
    va_rows = np.asarray(split.get('val', []), dtype=np.int64)
    te_rows = np.asarray(split['test'], dtype=np.int64)

    in_tr = torch.from_numpy(np.isin(obs_rows, tr_rows))                           # (m,)
    in_va = torch.from_numpy(np.isin(obs_rows, va_rows)) if va_rows.size else torch.zeros_like(in_tr, dtype=torch.bool)
    in_te = torch.from_numpy(np.isin(obs_rows, te_rows))

    # === 組裝 data 物件（與 UCI loader 對齊）===
    data = SN()

    # 供匯出/形狀參考
    data.df_X = X                                  # numpy array 就好
    data.num_obs  = int(N)
    data.num_feat = int(d)
    data.num_nodes = int(N + d)

    # 節點特徵：GRAPE 允許沒有真正 node feature，但為避免 None 觸發屬性錯，
    # 提供一維常數特徵即可
    data.x = torch.zeros((N + d, 1), dtype=torch.float32)
    data.num_node_features = 1

    # 邊（全部；給 gnn_y 內部重建 X 用）
    data.edge_index = edge_index_all
    data.edge_attr_dim = int(edge_attr_all.shape[1])  # = 1

    # 訓練邊（雙向）
    data.train_edge_index = torch.cat([edge_index_fwd[:, in_tr], edge_index_rev[:, in_tr]], dim=1)
    data.train_edge_attr  = torch.cat([edge_attr_fwd[in_tr],     edge_attr_rev[in_tr]],     dim=0)

    # 測試邊（雙向）
    data.test_edge_index  = torch.cat([edge_index_fwd[:, in_te], edge_index_rev[:, in_te]], dim=1)
    data.test_edge_attr   = torch.cat([edge_attr_fwd[in_te],     edge_attr_rev[in_te]],     dim=0)

    # MDI 標籤（僅取 forward 半邊；長度需等於 forward 邊數）
    data.train_labels = edge_attr_fwd[in_tr][:, 0]    # (m_train,)
    data.test_labels  = edge_attr_fwd[in_te][:, 0]    # (m_test,)

    # y 監督與 row 級 masks（給 gnn_y 用）
    data.y = torch.from_numpy(y.astype(np.float32))
    idx_all = np.arange(N, dtype=np.int64)
    data.train_y_mask = torch.from_numpy(np.isin(idx_all, tr_rows))
    data.valid_y_mask = torch.from_numpy(np.isin(idx_all, va_rows)) if va_rows.size else torch.zeros(N, dtype=torch.bool)
    data.test_y_mask  = torch.from_numpy(np.isin(idx_all, te_rows))

    return data
