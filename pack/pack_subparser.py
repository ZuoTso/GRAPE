from argparse import ArgumentParser

def add_pack_subparser(subparsers: 'ArgumentParser.add_subparsers'):
    """在 train_y.py / train_mdi.py 裡註冊 'pack' 子命令。
    讓你可以這樣用：
        python train_y.py pack --root /path/to/baseline/year_mcar30 --data year_mcar30
        python train_mdi.py pack --root /path/to/baseline/year_mcar30 --data year_mcar30
    """
    p = subparsers.add_parser('pack', help='Load pre-packed baseline folder (X_norm.npy, y.npy, mask.npy, split_idx.json)')
    p.add_argument('--root', required=True, help='Path to baseline folder produced by t2g_to_baseline.py')
    p.add_argument('--data', default='custom', help='Dataset name for logs (default: custom)')
    p.set_defaults(domain='pack')   # ← 加這一行，確保 args.domain 存在
    # 其餘通用參數交由主 parser（例如 seed、hidden_dim…）處理
    return p
