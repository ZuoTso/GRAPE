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

    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        predict_model.train()

        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
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