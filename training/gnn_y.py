import numpy as np
import torch
import torch.nn.functional as F
import pickle

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge

def train_gnn_y(data, args, log_path, device=torch.device('cpu')):
    model = get_gnn(data, args).to(device)

    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int, args.impute_hiddens.split('_')))
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2
    impute_model = MLPNet(input_dim, 1,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)

    if args.predict_hiddens == '':
        predict_hiddens = []
    else:
        predict_hiddens = list(map(int, args.predict_hiddens.split('_')))
    n_row, n_col = data.df_X.shape
    predict_model = MLPNet(n_col, 1,
                           hidden_layer_sizes=predict_hiddens,
                           dropout=args.dropout).to(device)

    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters()) \
                           + list(predict_model.parameters())

    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    Train_loss = []
    Test_rmse = []
    Test_l1 = []
    Lr = []

    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    edge_index = data.edge_index.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)
    if args.valid > 0.:
        valid_mask = get_known_mask(args.valid, all_train_y_mask.shape[0]).to(device)
        valid_mask = valid_mask*all_train_y_mask
        train_y_mask = all_train_y_mask.clone().detach()
        train_y_mask[valid_mask] = False
        valid_y_mask = all_train_y_mask.clone().detach()
        valid_y_mask[~valid_mask] = False
        print("all y num is {}, train num is {}, valid num is {}, test num is {}"\
                .format(
                all_train_y_mask.shape[0],torch.sum(train_y_mask),
                torch.sum(valid_y_mask),torch.sum(test_y_mask)))
        Valid_rmse = []
        Valid_l1 = []
        best_valid_rmse = np.inf
        best_valid_rmse_epoch = 0
        best_valid_l1 = np.inf
        best_valid_l1_epoch = 0
    else:
        train_y_mask = all_train_y_mask.clone().detach()
        print("all y num is {}, train num is {}, test num is {}"\
                .format(
                all_train_y_mask.shape[0],torch.sum(train_y_mask),
                torch.sum(test_y_mask)))

    # === EXPORT (gnn_y) — place just before training loop ===
    def _artifact_flags(args=None):
        import os
        art  = getattr(args, "artifact_dir", None) or os.getenv("GRAPT_ARTIFACT_DIR")
        dump = getattr(args, "dump_intermediate", False) or (os.getenv("GRAPT_DUMP_INTERMEDIATE") == "1")
        prep = getattr(args, "prep_only", False) or (os.getenv("GRAPT_PREP_ONLY") == "1")
        return art, dump, prep

    artifact_dir, dump_inter, prep_only = _artifact_flags(args if "args" in locals() else None)
    if artifact_dir and dump_inter:
        import os, json, numpy as np
        from utils.utils import save_baseline_artifacts, save_bipartite_edges

        save_dir = os.path.join(artifact_dir, "baseline", args.data, f"seed{args.seed}")
        n_row, n_col = data.df_X.shape

        # (A) 由【訓練輸入邊】鋪 mask：1=可見(訓練可用)、0=缺失
        ei = train_edge_index.detach().cpu().numpy()      # (2, E_train*2)
        rows = ei[0].astype(np.int64)
        cols = (ei[1] - n_row).astype(np.int64)          # 特徵節點以 n_row 偏移
        ok   = (rows>=0)&(rows<n_row)&(cols>=0)&(cols<n_col)
        mask_np = np.zeros((n_row, n_col), dtype=np.uint8)
        mask_np[rows[ok], cols[ok]] = 1

        # (B) X（沿用 data.df_X；多數資料已歸一/標準化）
        X_np = np.asarray(data.df_X, dtype=np.float32)

        # (C) row 級 split（沿用你原本的 y-masks）
        tr_idx = np.where(train_y_mask.detach().cpu().numpy())[0].tolist()
        te_idx = np.where(test_y_mask.detach().cpu().numpy())[0].tolist()
        va_idx = (np.where(valid_y_mask.detach().cpu().numpy())[0].tolist()
                if (hasattr(args,'valid') and args.valid>0.) else [])
        with open(os.path.join(save_dir, "split_idx.json"), "w") as f:
            json.dump({"train": tr_idx, "val": va_idx, "test": te_idx}, f)

        # (D) 二部圖（訓練可見邊；權重若無就全 1）
        if "train_edge_attr" in locals() and train_edge_attr is not None:
            w = train_edge_attr.detach().cpu().numpy().astype(np.float32)
            w = w[:rows.shape[0]]
        else:
            w = np.ones(rows.shape[0], dtype=np.float32)
        save_bipartite_edges(os.path.join(save_dir, "bipartite_edges.npz"),
                            rows[ok], cols[ok], w[ok])

        # (E) y（供後續對照）
        y_np = data.y.detach().cpu().numpy() if hasattr(data, "y") else None

        save_baseline_artifacts(save_dir, X_np, mask_np, y=y_np,
                                feature_names=getattr(data, "feature_names", None),
                                scaler=None,
                                split_idx={"train": tr_idx, "val": va_idx, "test": te_idx})

        if prep_only:
            print("[prep_only] exported intermediates; exit before training.")
            import sys; sys.exit(0)
    # === END EXPORT (gnn_y) ===

    # === Paste into gnn_y.py — 放在 `# === END EXPORT (gnn_y) ===` 之後、訓練迴圈前 ===
    # 目的（嚴格模式）：
    # 1) **必須**由 orchestrator 提供 manifest（GRAFT_OVERLAY_MANIFEST），且 **必須指定 masks 串接方式**。
    # 2) 若未指定 `mask_op`（AND/OR）、未提供 `masks`、或形狀不符，**立即報錯**，避免誤吃到舊檔。
    # 3) 按指定順序載入 masks（manifest.masks），依 `mask_op` 疊合，並可疊加 edge_keep_*。
    # 4) 僅過濾訓練邊與監督 row（軟刪）。硬刪位以 TODO[HARD_PRUNE] 註記。

    try:
        import os, sys, json, numpy as np
        from pathlib import Path

        man_path = os.getenv("GRAFT_OVERLAY_MANIFEST", "")
        if not man_path or not os.path.exists(man_path):
            raise RuntimeError("[FATAL gnn_y] Missing GRAFT_OVERLAY_MANIFEST (manifest not found). Orchestrator must specify it.")

        try:
            man = json.load(open(man_path, "r", encoding="utf-8"))
        except Exception as _e:
            raise RuntimeError(f"[FATAL gnn_y] Cannot parse manifest: {man_path}: {_e}")

        masks_paths = man.get("masks", [])
        edge_keep_paths = man.get("edge_keeps", [])
        order = man.get("order", None)
        # 串接方式：允許 env 覆寫，否則取 manifest.mask_op；**缺少就報錯**
        mask_op = os.getenv("GRAFT_MASK_OP", "").strip().upper() or str(man.get("mask_op", "")).strip().upper()
        if not masks_paths:
            raise RuntimeError("[FATAL gnn_y] overlay manifest has no 'masks'. Please list masks in order to compose.")
        if mask_op not in {"AND", "OR"}:
            raise RuntimeError("[FATAL gnn_y] mask_op not specified or invalid. Set manifest.mask_op or env GRAFT_MASK_OP to 'AND' or 'OR'.")
        if order is None:
            raise RuntimeError("[FATAL gnn_y] overlay manifest missing 'order'. Orchestrator must record module order (e.g., 't2g>lunar>grape').")

        n_row, n_col = data.df_X.shape

        # 疊 mask（按 manifest.masks 順序；op=AND/OR）
        def _infer_stage_from_path(p: str) -> str:
            name = Path(p).name
            if name == "mask.npy": return "baseline"
            if "mask_lunar" in name: return "lunar"
            if "mask_t2g" in name: return "t2g"
            if "mask_random" in name: return "random"
            return name  # fallback

        def _mask_stats(m: np.ndarray):
            ones = int(m.sum(dtype=np.int64))
            total = m.size
            cov = ones / total if total else 0.0
            row_keep = int(m.any(axis=1).sum())
            col_keep = int(m.any(axis=0).sum())
            return ones, cov, row_keep, col_keep

        DEBUG_MASK = os.getenv("GRAFT_DEBUG_MASK", "1") == "1"  # 想靜音就設 0

        if DEBUG_MASK:
            print(f"[overlay] mask_op={mask_op} | total_masks={len(masks_paths)} | "
                f"n_row={n_row}, n_col={n_col}")
            try:
                # 有的程式段會在上面讀過 manifest 的 'order'
                print(f"[overlay] order={order}")
            except NameError:
                pass

        M = None
        for i, p in enumerate(masks_paths, 1):
            try:
                m = np.load(p)
            except Exception as _e:
                raise RuntimeError(f"[FATAL gnn_y] cannot load mask: {p}: {_e}")
            if m.shape != (n_row, n_col):
                raise RuntimeError(f"[FATAL gnn_y] mask shape mismatch: {p} has {m.shape}, expected {(n_row, n_col)}")
            m = m.astype(np.uint8)
            stage = _infer_stage_from_path(p)

            if DEBUG_MASK:
                ones, cov, r_keep, c_keep = _mask_stats(m)
                print(f"[overlay:{i}] stage={stage:<8} op={mask_op:<3} path={p}")
                print(f"    current m: ones={ones} ({cov:.2%}) | rows_keep={r_keep}/{n_row} | cols_keep={c_keep}/{n_col}")

            if M is None:
                M = m
                if DEBUG_MASK:
                    onesA, covA, rA, cA = _mask_stats(M)
                    print(f"    init   M: ones={onesA} ({covA:.2%}) | rows_keep={rA}/{n_row} | cols_keep={cA}/{n_col}")
            else:
                onesB, covB, rB, cB = _mask_stats(M)
                M = (M & m) if mask_op == "AND" else (M | m)
                onesC, covC, rC, cC = _mask_stats(M)
                if DEBUG_MASK:
                    op_sym = "&" if mask_op == "AND" else "|"
                    print(f"    merge  M: prev_ones={onesB} ({covB:.2%}) {op_sym} current -> "
                        f"after_ones={onesC} ({covC:.2%}) | Δones={onesC - onesB:+d}, "
                        f"Δrows={rC - rB:+d}, Δcols={cC - cB:+d}")

        # （僅 gnn_y.py）若你在後面做 row_keep 過濾，也記得補一行摘要：
        if DEBUG_MASK:
            row_keep_np = M.any(axis=1)
            print(f"[overlay:final] rows_kept={int(row_keep_np.sum())}/{n_row} | "
                f"cols_kept={int(M.any(axis=0).sum())}/{n_col} | ones={int(M.sum(dtype=np.int64))}")


        # 1) 過濾訓練邊（只動 train_*，不動測試集合）
        ei = train_edge_index.detach().cpu().numpy()  # 2 x E
        keep = np.ones(ei.shape[1], dtype=bool)
        for j in range(ei.shape[1]):
            u = int(ei[0, j]); v = int(ei[1, j])
            if (u < n_row) ^ (v < n_row):
                r = u if u < n_row else v
                c = v - n_row if u < n_row else u - n_row
                if r >= n_row or c >= n_col or M[r, c] == 0:
                    keep[j] = False

        # 疊 edge_keep_*（若 manifest 指定；固定 AND 疊加；若要自定 op 可新增 edge_keep_op）
        if edge_keep_paths:
            k_all = None
            for kp in edge_keep_paths:
                if not os.path.exists(kp):
                    raise RuntimeError(f"[FATAL gnn_y] edge_keep file not found: {kp}")
                kv = np.load(kp).astype(bool)
                if kv.size != keep.size:
                    raise RuntimeError(f"[FATAL gnn_y] edge_keep size mismatch: {kp} ({kv.size} != {keep.size})")
                k_all = kv if k_all is None else (k_all & kv)
            keep &= k_all

        if (~keep).any():
            keep_t = torch.as_tensor(keep, device=train_edge_index.device)
            train_edge_index = train_edge_index[:, keep_t]
            if train_edge_attr is not None:
                train_edge_attr = train_edge_attr[keep_t]

        # 2) 過濾監督 row（排除任何欄位都不可見的樣本）
        row_keep_np = M.any(axis=1)
        row_keep = torch.as_tensor(row_keep_np, device=all_train_y_mask.device)
        all_train_y_mask = all_train_y_mask & row_keep
        test_y_mask      = test_y_mask      & row_keep
        if getattr(args, 'valid', 0.) > 0.:
            train_y_mask = train_y_mask & row_keep
            valid_y_mask = valid_y_mask & row_keep

        print(f"[IMPORT] gnn_y: mask_op={mask_op}; edges kept {int(keep.sum())}/{keep.size}; rows kept {int(row_keep_np.sum())}/{n_row}; order={order}")

        # ========================== 硬刪（未來擴充位） ==========================
        # TODO[HARD_PRUNE]: 建議由 orchestrator 的 finalize 階段先物化縮小版（X/mask/edges），
        # 並輸出 kept_rows/kept_cols；若要在此硬刪，需進行 reindex 與評估索引校驗。
        # ======================================================================

    except Exception as e:
        # 直接拋出以中止本次訓練，防止吃到錯誤組合
        raise
    # === END Paste into gnn_y.py — 放在 `# === END EXPORT (gnn_y) ===` 之後、訓練迴圈前 ===

    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        predict_model.train()

        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        # ================================== pack ====================================
        if getattr(args, "domain", "uci") == "pack":
            # 產生 rows×cols 的網格索引（row 節點是 [0..n_row-1]，col 節點是 [n_row..n_row+n_col-1]）
            row_idx = torch.arange(n_row, device=device).repeat_interleave(n_col)
            col_idx = (torch.arange(n_col, device=device) + n_row).repeat(n_row)
            X = impute_model([x_embd[row_idx], x_embd[col_idx]])
            X = X.view(n_row, n_col)
        else:
        # ================================== pack ====================================
            X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
            X = torch.reshape(X, [n_row, n_col])

        pred = predict_model(X)[:, 0]
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]

        loss = F.mse_loss(pred_train, label_train)
        loss.backward()
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])

        model.eval()
        impute_model.eval()
        predict_model.eval()
        with torch.no_grad():
            if args.valid > 0.:
                x_embd = model(x, train_edge_attr, train_edge_index)
                # ================================== pack ====================================
                if getattr(args, "domain", "uci") == "pack":
                    # 產生 rows×cols 的網格索引（row 節點是 [0..n_row-1]，col 節點是 [n_row..n_row+n_col-1]）
                    row_idx = torch.arange(n_row, device=device).repeat_interleave(n_col)
                    col_idx = (torch.arange(n_col, device=device) + n_row).repeat(n_row)
                    X = impute_model([x_embd[row_idx], x_embd[col_idx]])
                    X = X.view(n_row, n_col)
                else:
                # ================================== pack ====================================
                    X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
                    X = torch.reshape(X, [n_row, n_col])
                pred = predict_model(X)[:, 0]
                pred_valid = pred[valid_y_mask]
                label_valid = y[valid_y_mask]
                mse = F.mse_loss(pred_valid, label_valid)
                valid_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_valid, label_valid)
                valid_l1 = l1.item()
                if valid_l1 < best_valid_l1:
                    best_valid_l1 = valid_l1
                    best_valid_l1_epoch = epoch
                    torch.save(model, log_path + 'model_best_valid_l1.pt')
                    torch.save(impute_model, log_path + 'impute_model_best_valid_l1.pt')
                    torch.save(predict_model, log_path + 'predict_model_best_valid_l1.pt')
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_valid_rmse_epoch = epoch
                    torch.save(model, log_path + 'model_best_valid_rmse.pt')
                    torch.save(impute_model, log_path + 'impute_model_best_valid_rmse.pt')
                    torch.save(predict_model, log_path + 'predict_model_best_valid_rmse.pt')
                Valid_rmse.append(valid_rmse)
                Valid_l1.append(valid_l1)

            x_embd = model(x, train_edge_attr, train_edge_index)
            # ================================== pack ====================================
            if getattr(args, "domain", "uci") == "pack":
                # 產生 rows×cols 的網格索引（row 節點是 [0..n_row-1]，col 節點是 [n_row..n_row+n_col-1]）
                row_idx = torch.arange(n_row, device=device).repeat_interleave(n_col)
                col_idx = (torch.arange(n_col, device=device) + n_row).repeat(n_row)
                X = impute_model([x_embd[row_idx], x_embd[col_idx]])
                X = X.view(n_row, n_col)
            else:
            # ================================== pack ====================================
                X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
                X = torch.reshape(X, [n_row, n_col])
            pred = predict_model(X)[:, 0]
            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]
            mse = F.mse_loss(pred_test, label_test)
            test_rmse = np.sqrt(mse.item())
            l1 = F.l1_loss(pred_test, label_test)
            test_l1 = l1.item()

            Train_loss.append(train_loss)
            Test_rmse.append(test_rmse)
            Test_l1.append(test_l1)
            print('epoch: ', epoch)
            print('loss: ', train_loss)
            if args.valid > 0.:
                print('valid rmse: ', valid_rmse)
                print('valid l1: ', valid_l1)
            print('test rmse: ', test_rmse)
            print('test l1: ', test_l1)

    # --- for graft artifact
    # 訓練回圈後（已有 test_rmse/test_l1/pred_test/label_test），追加：
    if (getattr(args, "artifact_dir", None) or os.getenv("GRAPT_ARTIFACT_DIR")) and \
        (getattr(args, "dump_intermediate", False) or os.getenv("GRAPT_DUMP_INTERMEDIATE") == "1"):
        out_dir = os.path.join(args.artifact_dir, "baseline", args.data, f"seed{args.seed}", "label")
        os.makedirs(out_dir, exist_ok=True)

        # y_pred（僅測試列；如需全域，可另外導出）
        np.save(os.path.join(out_dir, "y_pred.npy"),    pred_test.detach().cpu().numpy())
        np.save(os.path.join(out_dir, "label_test.npy"), label_test.detach().cpu().numpy())

        # 指標（與 GRAPE 論文一致：RMSE/MAE；可視需要再加 MSE/AUC）
        metrics = {
            "test": {"rmse": float(test_rmse), "mae": float(test_l1)},
        }
        if args.valid > 0.:
            metrics["valid"] = {"rmse": float(best_valid_rmse), "mae": float(best_valid_l1)}
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
    # ---
    
    pred_train = pred_train.detach().cpu().numpy()
    label_train = label_train.detach().cpu().numpy()
    pred_test = pred_test.detach().cpu().numpy()
    label_test = label_test.detach().cpu().numpy()

    obj = dict()
    obj['args'] = args
    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    if args.valid > 0.:
        obj['curves']['valid_rmse'] = Valid_rmse
        obj['curves']['valid_l1'] = Valid_l1
    obj['curves']['test_rmse'] = Test_rmse
    obj['curves']['test_l1'] = Test_l1
    obj['lr'] = Lr
    obj['outputs'] = dict()
    obj['outputs']['pred_train'] = pred_train
    obj['outputs']['label_train'] = label_train
    obj['outputs']['pred_test'] = pred_test
    obj['outputs']['label_test'] = label_test
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    torch.save(model, log_path + 'model.pt')
    torch.save(impute_model, log_path + 'impute_model.pt')
    torch.save(predict_model, log_path + 'predict_model.pt')

    # obj = objectview(obj)
    plot_curve(obj['curves'], log_path+'curves.png',keys=None, 
                clip=True, label_min=True, label_end=True)
    plot_curve(obj, log_path+'lr.png',keys=['lr'], 
                clip=False, label_min=False, label_end=False)
    plot_sample(obj['outputs'], log_path+'outputs.png', 
                groups=[['pred_train','label_train'],
                        ['pred_test','label_test']
                        ], 
                num_points=20)
    if args.valid > 0.:
        print("best valid rmse is {:.3g} at epoch {}".format(best_valid_rmse,best_valid_rmse_epoch))
        print("best valid l1 is {:.3g} at epoch {}".format(best_valid_l1,best_valid_l1_epoch))