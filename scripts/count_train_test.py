from pathlib import Path
import unicodedata


def get_display_width(text):
    """
    计算字符串在终端中的显示宽度：
    中文、全角字符计为 2；
    英文、数字、半角符号计为 1。
    """
    text = str(text)
    width = 0

    for char in text:
        if unicodedata.east_asian_width(char) in ("W", "F"):
            width += 2
        else:
            width += 1

    return width


def pad_text(text, total_width, align="left"):
    """
    根据显示宽度进行填充。
    align:
        left  左对齐
        right 右对齐
        center 居中
    """
    text = str(text)
    curr_width = get_display_width(text)
    padding = max(0, total_width - curr_width)

    if align == "right":
        return " " * padding + text
    elif align == "center":
        left = padding // 2
        right = padding - left
        return " " * left + text + " " * right
    else:
        return text + " " * padding


def make_line(values, widths, aligns):
    """生成表格中的一行"""
    cells = []

    for value, width, align in zip(values, widths, aligns):
        cells.append(" " + pad_text(value, width, align) + " ")

    return "|" + "|".join(cells) + "|"


def make_separator(widths):
    """生成表格分隔线"""
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"


def count_and_log():
    # 路径定位
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    train_root = project_root / "data" / "train_runs" / "dataset"
    test_root = project_root / "data" / "test_runs" / "dataset"

    # 类别映射表
    category_map = {
        "02691156": "飞机",
        "02747177": "垃圾桶",
        "02773838": "包",
        "02801938": "篮子",
        "02808440": "浴缸",
        "02818832": "床",
        "02828884": "长凳",
        "02843684": "鸟屋",
        "02871439": "书架",
        "02876657": "瓶子",
        "02880940": "碗",
        "02924116": "公交车",
        "02933112": "柜子",
        "02942699": "相机",
        "02946921": "罐子",
        "02954340": "帽子",
        "02958343": "汽车",
        "02992529": "椅子",
        "03001627": "时钟",
        "03046257": "时钟(变体)",
        "03085013": "未知类别",
        "03207941": "洗碗机",
        "03211117": "显示器",
        "03261776": "耳机",
        "03325088": "水龙头",
        "03337140": "文件柜",
        "03467517": "吉他",
        "03513137": "头盔",
        "03593526": "罐子(Jar)",
        "03624134": "刀",
        "03636649": "灯",
        "03642806": "笔记本电脑",
        "03691459": "扬声器",
        "03710193": "信箱",
        "03759954": "麦克风",
        "03761084": "微波炉",
        "03790512": "摩托车",
        "03797390": "杯子",
        "03928116": "钢琴",
        "03938244": "枕头",
        "03948459": "手枪",
        "03991062": "锅",
        "04004475": "打印机",
        "04074963": "遥控器",
        "04090263": "步枪",
        "04099429": "火箭",
        "04225987": "滑板",
        "04256520": "沙发",
        "04330267": "炉子",
        "04379243": "桌子",
        "04401088": "电话",
        "04460130": "塔",
        "04468005": "未知类别",
        "04530566": "未知类别",
        "04554684": "洗衣机",
    }

    if not train_root.exists() and not test_root.exists():
        print(f"❌ 路径不存在:")
        print(f"训练集路径: {train_root}")
        print(f"测试集路径: {test_root}")
        return

    train_categories = {
        d.name for d in train_root.iterdir() if d.is_dir()
    } if train_root.exists() else set()

    test_categories = {
        d.name for d in test_root.iterdir() if d.is_dir()
    } if test_root.exists() else set()

    all_categories = sorted(train_categories | test_categories)

    # 表格列宽
    widths = [15, 20, 12, 12]

    # 对齐方式
    aligns = ["left", "left", "right", "right"]

    header = ["文件夹名", "中文名", "训练集", "测试集"]
    separator = make_separator(widths)

    total_train = 0
    total_test = 0

    print("\n[📊 数据集分布统计]")
    print(separator)
    print(make_line(header, widths, ["center", "center", "center", "center"]))
    print(separator)

    for cat in all_categories:
        train_cat_path = train_root / cat
        test_cat_path = test_root / cat

        train_count = (
            sum(1 for d in train_cat_path.iterdir() if d.is_dir())
            if train_cat_path.exists()
            else 0
        )

        test_count = (
            sum(1 for d in test_cat_path.iterdir() if d.is_dir())
            if test_cat_path.exists()
            else 0
        )

        total_train += train_count
        total_test += test_count

        chinese_name = category_map.get(cat, "未知")

        row = [
            cat,
            chinese_name,
            train_count,
            test_count,
        ]

        print(make_line(row, widths, aligns))

    print(separator)

    # 最后一行统计总数
    total_row = [
        "合计",
        f"{len(all_categories)} 个类别",
        total_train,
        total_test,
    ]

    print(make_line(total_row, widths, aligns))
    print(separator)


if __name__ == "__main__":
    count_and_log()