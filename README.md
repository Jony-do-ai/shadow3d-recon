训练：在根目录执行
python scripts/train.py --config configs/train_pct.yaml


python scripts/infer.py --config data/train_runs/apml-shadow-2part2/config_dump.yaml `
    --checkpoint data/train_runs/apml-shadow-2part2/checkpoints/best.pt --all

python scripts/eval_existing_pcd_metrics_csv.py `
     --root outputs/infer/apml-shadow-2part `
     --thresholds 0.01 0.02 0.03 0.05 `
     --device cuda

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
| 02992529        | 椅子                 |          514 |           30 |
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
| 合计            | 23 个类别            |         5415 |          265 |
+-----------------+----------------------+--------------+--------------+