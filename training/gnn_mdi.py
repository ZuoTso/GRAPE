import numpy as np
import torch
import torch.nn.functional as F
import pickle

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge

# ===
def to_numpy(x):
    import numpy as np
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)
# ===

def train_gnn_mdi(data, args, log_path, device=torch.device('cpu')):
    model = get_gnn(data, args).to(device)
    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int,args.impute_hiddens.split('_')))
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2
    if hasattr(args,'ce_loss') and args.ce_loss:
        output_dim = len(data.class_values)
    else:
        output_dim = 1
    impute_model = MLPNet(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)
    if args.transfer_dir: # this ensures the valid mask is consistant
        load_path = './{}/test/{}/{}/'.format(args.domain,args.data,args.transfer_dir)
        print("loading fron {} with {}".format(load_path,args.transfer_extra))
        model = torch.load(load_path+'model'+args.transfer_extra+'.pt',map_location=device)
        impute_model = torch.load(load_path+'impute_model'+args.transfer_extra+'.pt',map_location=device)

    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters())
    print("total trainable_parameters: ",len(trainable_parameters))
    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    Train_loss = []
    Test_rmse = []
    Test_l1 = []
    Lr = []

    x = data.x.clone().detach().to(device)
    if hasattr(args,'split_sample') and args.split_sample > 0.:
        if args.split_train:
            all_train_edge_index = data.lower_train_edge_index.clone().detach().to(device)
            all_train_edge_attr = data.lower_train_edge_attr.clone().detach().to(device)
            all_train_labels = data.lower_train_labels.clone().detach().to(device)
        else:
            all_train_edge_index = data.train_edge_index.clone().detach().to(device)
            all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
            all_train_labels = data.train_labels.clone().detach().to(device)
        if args.split_test:
            test_input_edge_index = data.higher_train_edge_index.clone().detach().to(device)
            test_input_edge_attr = data.higher_train_edge_attr.clone().detach().to(device)
        else:
            test_input_edge_index = data.train_edge_index.clone().detach().to(device)
            test_input_edge_attr = data.train_edge_attr.clone().detach().to(device)
        test_edge_index = data.higher_test_edge_index.clone().detach().to(device)
        test_edge_attr = data.higher_test_edge_attr.clone().detach().to(device)
        test_labels = data.higher_test_labels.clone().detach().to(device)
    else:
        all_train_edge_index = data.train_edge_index.clone().detach().to(device)
        all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
        all_train_labels = data.train_labels.clone().detach().to(device)
        test_input_edge_index = all_train_edge_index
        test_input_edge_attr = all_train_edge_attr
        test_edge_index = data.test_edge_index.clone().detach().to(device)
        test_edge_attr = data.test_edge_attr.clone().detach().to(device)
        test_labels = data.test_labels.clone().detach().to(device)
    if hasattr(data,'class_values'):
        class_values = data.class_values.clone().detach().to(device)
    if args.valid > 0.:
        valid_mask = get_known_mask(args.valid, int(all_train_edge_attr.shape[0] / 2)).to(device)
        print("valid mask sum: ",torch.sum(valid_mask))
        train_labels = all_train_labels[~valid_mask]
        valid_labels = all_train_labels[valid_mask]
        double_valid_mask = torch.cat((valid_mask, valid_mask), dim=0)
        valid_edge_index, valid_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, double_valid_mask, True)
        train_edge_index, train_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, ~double_valid_mask, True)
        print("train edge num is {}, valid edge num is {}, test edge num is input {} output {}"\
                .format(
                train_edge_attr.shape[0], valid_edge_attr.shape[0],
                test_input_edge_attr.shape[0], test_edge_attr.shape[0]))
        Valid_rmse = []
        Valid_l1 = []
        best_valid_rmse = np.inf
        best_valid_rmse_epoch = 0
        best_valid_l1 = np.inf
        best_valid_l1_epoch = 0
    else:
        train_edge_index, train_edge_attr, train_labels =\
             all_train_edge_index, all_train_edge_attr, all_train_labels
        print("train edge num is {}, test edge num is input {}, output {}"\
                .format(
                train_edge_attr.shape[0],
                test_input_edge_attr.shape[0], test_edge_attr.shape[0]))
    if args.auto_known:
        args.known = float(all_train_labels.shape[0])/float(all_train_labels.shape[0]+test_labels.shape[0])
        print("auto calculating known is {}/{} = {:.3g}".format(all_train_labels.shape[0],all_train_labels.shape[0]+test_labels.shape[0],args.known))
    obj = dict()
    obj['args'] = args
    obj['outputs'] = dict()

    # === EXPORT (gnn_mdi) — place just before training loop ===
    def _artifact_flags(args=None):
        import os
        art  = getattr(args, "artifact_dir", None) or os.getenv("GRAPT_ARTIFACT_DIR")
        dump = getattr(args, "dump_intermediate", False) or (os.getenv("GRAPT_DUMP_INTERMEDIATE") == "1")
        prep = getattr(args, "prep_only", False) or (os.getenv("GRAPT_PREP_ONLY") == "1")
        return art, dump, prep

    artifact_dir, dump_inter, prep_only = _artifact_flags(args if "args" in locals() else None)
    if artifact_dir and dump_inter:
        import os, numpy as np
        from utils.utils import save_baseline_artifacts, save_bipartite_edges, save_omega_test

        save_dir = os.path.join(args.artifact_dir, "baseline", args.data, f"seed{args.seed}")
        n_row, n_col = data.df_X.shape

        # (A) 由【訓練輸入邊】鋪 mask（優先 all_train_*，否則 train_*）
        ei_train = all_train_edge_index if "all_train_edge_index" in locals() else train_edge_index
        ea_train = all_train_edge_attr  if "all_train_edge_attr"  in locals() else train_edge_attr
        ei = ei_train.detach().cpu().numpy()
        r = ei[0].astype(np.int64)
        c = (ei[1] - n_row).astype(np.int64)
        ok = (r>=0)&(r<n_row)&(c>=0)&(c<n_col)
        mask_np = np.zeros((n_row, n_col), dtype=np.uint8)
        mask_np[r[ok], c[ok]] = 1

        # (B) X：沿用 data.df_X
        X_np = np.asarray(data.df_X, dtype=np.float32)

        # (C) 二部圖（訓練可見邊）
        if ea_train is not None:
            w = ea_train.detach().cpu().numpy().astype(np.float32)
            w = w[:r.shape[0]]
        else:
            w = np.ones(r.shape[0], dtype=np.float32)
        save_bipartite_edges(os.path.join(save_dir, "bipartite_edges.npz"), r[ok], c[ok], w[ok])

        # (D) Ω：由 test_edge_index 轉 row/col（固定為之後公平評估用）
        te = test_edge_index.detach().cpu().numpy()
        omega_rows = te[0].astype(np.int64)
        omega_cols = (te[1] - n_row).astype(np.int64)
        ok2 = (omega_rows>=0)&(omega_rows<n_row)&(omega_cols>=0)&(omega_cols<n_col)
        save_omega_test(os.path.join(save_dir, "omega_test_idx.npz"), omega_rows[ok2], omega_cols[ok2])

        # (E) 寫出 X/mask（impute 任務沒有 row 級 y）
        save_baseline_artifacts(save_dir, X_np, mask_np, y=None,
                                feature_names=getattr(data, "feature_names", None),
                                scaler=None, split_idx=None)

        if prep_only:
            print("[prep_only] exported intermediates; exit before training.")
            import sys; sys.exit(0)
    # === END EXPORT (gnn_mdi) ===

    # === Paste into gnn_mdi.py — 放在 `# === END EXPORT (gnn_mdi) ===` 之後、訓練前 ===
    # 嚴格模式與 gnn_y 同；差異在於不需要 row 過濾（僅過濾訓練邊）。

    try:
        import os, sys, json, numpy as np

        man_path = os.getenv("GRAFT_OVERLAY_MANIFEST", "")
        if not man_path or not os.path.exists(man_path):
            raise RuntimeError("[FATAL gnn_mdi] Missing GRAFT_OVERLAY_MANIFEST (manifest not found). Orchestrator must specify it.")

        try:
            man = json.load(open(man_path, "r", encoding="utf-8"))
        except Exception as _e:
            raise RuntimeError(f"[FATAL gnn_mdi] Cannot parse manifest: {man_path}: {_e}")

        masks_paths = man.get("masks", [])
        edge_keep_paths = man.get("edge_keeps", [])
        order = man.get("order", None)
        mask_op = os.getenv("GRAFT_MASK_OP", "").strip().upper() or str(man.get("mask_op", "")).strip().upper()
        if not masks_paths:
            raise RuntimeError("[FATAL gnn_mdi] overlay manifest has no 'masks'. Please list masks in order to compose.")
        if mask_op not in {"AND", "OR"}:
            raise RuntimeError("[FATAL gnn_mdi] mask_op not specified or invalid. Set manifest.mask_op or env GRAFT_MASK_OP to 'AND' or 'OR'.")
        if order is None:
            raise RuntimeError("[FATAL gnn_mdi] overlay manifest missing 'order'. Orchestrator must record module order (e.g., 't2g>lunar>grape').")

        n_row, n_col = data.df_X.shape

        M = None
        for p in masks_paths:
            try:
                m = np.load(p)
            except Exception as _e:
                raise RuntimeError(f"[FATAL gnn_mdi] cannot load mask: {p}: {_e}")
            if m.shape != (n_row, n_col):
                raise RuntimeError(f"[FATAL gnn_mdi] mask shape mismatch: {p} has {m.shape}, expected {(n_row, n_col)}")
            m = m.astype(np.uint8)
            if M is None:
                M = m
            else:
                M = (M & m) if mask_op == "AND" else (M | m)

        ei = train_edge_index.detach().cpu().numpy()
        keep = np.ones(ei.shape[1], dtype=bool)
        for j in range(ei.shape[1]):
            u = int(ei[0, j]); v = int(ei[1, j])
            if (u < n_row) ^ (v < n_row):
                r = u if u < n_row else v
                c = v - n_row if u < n_row else u - n_row
                if r >= n_row or c >= n_col or M[r, c] == 0:
                    keep[j] = False

        if edge_keep_paths:
            k_all = None
            for kp in edge_keep_paths:
                if not os.path.exists(kp):
                    raise RuntimeError(f"[FATAL gnn_mdi] edge_keep file not found: {kp}")
                kv = np.load(kp).astype(bool)
                if kv.size != keep.size:
                    raise RuntimeError(f"[FATAL gnn_mdi] edge_keep size mismatch: {kp} ({kv.size} != {keep.size})")
                k_all = kv if k_all is None else (k_all & kv)
            keep &= k_all

        if (~keep).any():
            keep_t = torch.as_tensor(keep, device=train_edge_index.device)
            train_edge_index = train_edge_index[:, keep_t]
            train_edge_attr  = train_edge_attr[keep_t]
            if 'test_input_edge_index' in locals() and test_input_edge_index.shape[1] == ei.shape[1]:
                test_input_edge_index = test_input_edge_index[:, keep_t]
                test_input_edge_attr  = test_input_edge_attr[keep_t]

        print(f"[IMPORT] gnn_mdi: mask_op={mask_op}; edges kept {int(keep.sum())}/{keep.size}; order={order}")

        # ========================== 硬刪（未來擴充位） ==========================
        # TODO[HARD_PRUNE]: 建議 orchestrator 先物化縮小版（X/mask/edges + kept_rows/kept_cols），
        # 本檔只讀縮小資料。若強行在此硬刪，需 reindex 訓練/驗證切分並重建邊，並校驗 omega_test 一致性。
        # ======================================================================

    except Exception as e:
        # 直接拋出以中止本次訓練
        raise
    # === END Paste into gnn_mdi.py — 放在 `# === END EXPORT (gnn_mdi) ===` 之後、訓練前 ===

    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
        if hasattr(args,'ce_loss') and args.ce_loss:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2)]
        else:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2),0]
        if args.loss_mode == 1:
            pred_train[known_mask] = train_labels[known_mask]
        label_train = train_labels

        if hasattr(args,'ce_loss') and args.ce_loss:
            loss = F.cross_entropy(pred_train,train_labels)
        else:
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
        with torch.no_grad():
            if args.valid > 0.:
                x_embd = model(x, train_edge_attr, train_edge_index)
                pred = impute_model([x_embd[valid_edge_index[0], :], x_embd[valid_edge_index[1], :]])
                if hasattr(args,'ce_loss') and args.ce_loss:
                    pred_valid = class_values[pred[:int(valid_edge_attr.shape[0] / 2)].max(1)[1]]
                    label_valid = class_values[valid_labels]
                elif hasattr(args,'norm_label') and args.norm_label:
                    pred_valid = pred[:int(valid_edge_attr.shape[0] / 2),0]
                    pred_valid = pred_valid * max(class_values)
                    label_valid = valid_labels
                    label_valid = label_valid * max(class_values)
                else:
                    pred_valid = pred[:int(valid_edge_attr.shape[0] / 2),0]
                    label_valid = valid_labels
                mse = F.mse_loss(pred_valid, label_valid)
                valid_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_valid, label_valid)
                valid_l1 = l1.item()
                if valid_l1 < best_valid_l1:
                    best_valid_l1 = valid_l1
                    best_valid_l1_epoch = epoch
                    if args.save_model:
                        torch.save(model, log_path + 'model_best_valid_l1.pt')
                        torch.save(impute_model, log_path + 'impute_model_best_valid_l1.pt')
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_valid_rmse_epoch = epoch
                    if args.save_model:
                        torch.save(model, log_path + 'model_best_valid_rmse.pt')
                        torch.save(impute_model, log_path + 'impute_model_best_valid_rmse.pt')
                Valid_rmse.append(valid_rmse)
                Valid_l1.append(valid_l1)

            x_embd = model(x, test_input_edge_attr, test_input_edge_index)
            pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
            if hasattr(args,'ce_loss') and args.ce_loss:
                pred_test = class_values[pred[:int(test_edge_attr.shape[0] / 2)].max(1)[1]]
                label_test = class_values[test_labels]
            elif hasattr(args,'norm_label') and args.norm_label:
                pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
                pred_test = pred_test * max(class_values)
                label_test = test_labels
                label_test = label_test * max(class_values)
            else:
                pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
                label_test = test_labels
            mse = F.mse_loss(pred_test, label_test)
            test_rmse = np.sqrt(mse.item())
            l1 = F.l1_loss(pred_test, label_test)
            test_l1 = l1.item()
            if args.save_prediction:
                if epoch == best_valid_rmse_epoch:
                    obj['outputs']['best_valid_rmse_pred_test'] = pred_test.detach().cpu().numpy()
                if epoch == best_valid_l1_epoch:
                    obj['outputs']['best_valid_l1_pred_test'] = pred_test.detach().cpu().numpy()

            if args.mode == 'debug':
                torch.save(model, log_path + 'model_{}.pt'.format(epoch))
                torch.save(impute_model, log_path + 'impute_model_{}.pt'.format(epoch))
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

    pred_train = pred_train.detach().cpu().numpy()
    label_train = label_train.detach().cpu().numpy()
    pred_test = pred_test.detach().cpu().numpy()
    label_test = label_test.detach().cpu().numpy()

    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    if args.valid > 0.:
        obj['curves']['valid_rmse'] = Valid_rmse
        obj['curves']['valid_l1'] = Valid_l1
    obj['curves']['test_rmse'] = Test_rmse
    obj['curves']['test_l1'] = Test_l1
    obj['lr'] = Lr

    obj['outputs']['final_pred_train'] = pred_train
    obj['outputs']['label_train'] = label_train
    obj['outputs']['final_pred_test'] = pred_test
    obj['outputs']['label_test'] = label_test
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    # --- for graft artifact
    # 在 gnn_mdi.py 訓練末端（pickle.dump(obj, ...) 前後皆可）追加：
    if (getattr(args, "artifact_dir", None) or os.getenv("GRAPT_ARTIFACT_DIR")) and \
        (getattr(args, "dump_intermediate", False) or os.getenv("GRAPT_DUMP_INTERMEDIATE") == "1"):
        out_dir = os.path.join(args.artifact_dir, "baseline", args.data, f"seed{args.seed}", "impute")
        os.makedirs(out_dir, exist_ok=True)

        import json
        # 邊級預測與標籤（以 test 邊集合）
        np.save(os.path.join(out_dir, "edge_pred.npy"),  to_numpy(pred_test))
        np.save(os.path.join(out_dir, "edge_label.npy"), to_numpy(label_test))

        # 指標：RMSE/MAE
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"test": {"rmse": float(test_rmse), "mae": float(test_l1)}}, f)
    # ---

    if args.save_model:
        torch.save(model, log_path + 'model.pt')
        torch.save(impute_model, log_path + 'impute_model.pt')

    # obj = objectview(obj)
    plot_curve(obj['curves'], log_path+'curves.png',keys=None, 
                clip=True, label_min=True, label_end=True)
    plot_curve(obj, log_path+'lr.png',keys=['lr'], 
                clip=False, label_min=False, label_end=False)
    plot_sample(obj['outputs'], log_path+'outputs.png', 
                groups=[['final_pred_train','label_train'],
                        ['final_pred_test','label_test']
                        ], 
                num_points=20)
    if args.save_prediction and args.valid > 0.:
        plot_sample(obj['outputs'], log_path+'outputs_best_valid.png', 
                    groups=[['best_valid_rmse_pred_test','label_test'],
                            ['best_valid_l1_pred_test','label_test']
                            ], 
                    num_points=20)
    if args.valid > 0.:
        print("best valid rmse is {:.3g} at epoch {}".format(best_valid_rmse,best_valid_rmse_epoch))
        print("best valid l1 is {:.3g} at epoch {}".format(best_valid_l1,best_valid_l1_epoch))
