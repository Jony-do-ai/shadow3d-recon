训练：在根目录执行
python scripts/train.py --config configs/train_default.yaml


python scripts/infer.py --config configs/train_default.yaml --checkpoint data/train_runs/shadow_point_baseline/checkpoints/best.pt --all

将训练集对应数量的数据移动到测试集
只列出转移信息用于查看，不做转移操作：
python src/shadow3d/utils/split_train_test.py --dry_run 
直接做转移操作：
python src/shadow3d/utils/split_train_test.py