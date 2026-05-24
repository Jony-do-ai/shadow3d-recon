训练：在根目录执行
python scripts/train.py --config configs/train_pct_apml.yaml

python scripts/train.py --config configs/train_pct_apml.yaml --resume outputs/train_runs/cnn-apml/checkpoints/epoch_0030.pt

python scripts/train.py `
  --config configs/train_pct_apml_projection_stage2.yaml `
  --resume outputs/train_runs/cnn-apml/checkpoints/epoch_0150.pt `
  --model_only_resume

推理
python scripts/infer.py --config configs/train_pct_apml_projection_stage2.yaml `
    --checkpoint outputs/train_runs/cnn-apml/checkpoints/best.pt --all

python scripts/infer.py --config configs/train_pct_apml_projection_stage2.yaml `
    --checkpoint outputs/train_runs/transformer-cnn-apmlcd/checkpoints/best.pt --all

推理指标
python scripts/eval_existing_pcd_metrics.py `
  --root outputs/infer/apml-nopct `
  --thresholds 0.01 0.02 0.03 0.05 `
  --device cuda

python scripts/eval_existing_pcd_metrics_csv.py `
    --root outputs/infer/drwr `
    --thresholds 0.01 0.02 0.03 0.05 `
    --device cuda

drwr的对角线=1的归一化转换并算cd
python scripts/drwr_cd_renormalize.py --root outputs/infer/drwr

敏感性分析
mask膨胀或者腐蚀
radius：自己设置像素值3，5，7，9；膨胀版：--op dilate，腐蚀版：--op erode
python src\shadow3d\utils\build_fuckpng.py --src data\test_runs\dataset --op erode --radius 30

帧顺序改变
光线和mask一起打包打乱
python src\shadow3d\utils\build_chaos_LS.py --src data\test_runs\dataset --seed 42
只mask打乱
python src\shadow3d\utils\build_chaos_S.py --src data\test_runs\dataset --seed 42


将训练集对应数量的数据移动到测试集
只列出转移信息用于查看，不做转移操作：
python scripts/split_train_test.py --dry_run 
直接做转移操作：
python scripts/split_train_test.py

数据集统计功能：
python scripts/count_train_test.py

数据图
python scripts/plot_train_log.py --log data/train_runs/exp_no_light/train_log.csv

[📊 数据集分布统计]
+-----------------+----------------------+--------------+--------------+
|    文件夹名     |        中文名        |    训练集    |    测试集    |
+-----------------+----------------------+--------------+--------------+
| 02747177        | 垃圾桶               |          323 |           20 |
| 02773838        | 包                   |           79 |            4 |
| 02801938        | 篮子                 |          108 |            5 |
| 02808440        | 浴缸                 |          826 |           30 |
| 02818832        | 床                   |          223 |           10 |
| 02843684        | 鸟屋                 |           69 |            4 |
| 02876657        | 瓶子                 |          478 |           20 |
| 02880940        | 碗                   |          178 |            8 |
| 02933112        | 柜子                 |          455 |           30 |
| 02946921        | 罐子                 |          103 |            5 |
| 02954340        | 帽子                 |           53 |            3 |
| 02992529        | 手机                 |          100 |           5 |
| 03261776        | 耳机                 |           69 |            4 |
| 03337140        | 文件柜               |          288 |           10 |
| 03513137        | 头盔                 |          154 |            8 |
| 03593526        | 罐子(Jar)            |          566 |           30 |
| 03636649        | 灯                   |          412 |           20 |
| 03710193        | 信箱                 |           90 |            4 |
| 03759954        | 麦克风               |           64 |            3 |
| 03938244        | 枕头                 |           92 |            4 |
| 04074963        | 遥控器               |           63 |            3 |
| 04099429        | 火箭                 |           81 |            4 |
| 04460130        | 塔                   |          127 |            6 |
+-----------------+----------------------+--------------+--------------+
| 合计            | 23 个类别            |         5001 |          265 |
+-----------------+----------------------+--------------+--------------+