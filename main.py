import subprocess
import io
from PIL import Image
import easyocr
import numpy as np
import os
import time
import math
from screenshot import Screenshot
import threading
import random
from queue import Queue

reader = easyocr.Reader(['en'], gpu=False)

# 截图器
screenshot_obj = Screenshot(200)

# 种植区域参数
PLANT_AREA_TOP_LEFT = (515, 150)
PLANT_AREA_SIZE = (1230, 820)
PLANT_AREA_GRID = (9, 5)

# 记录每个格子的植物占用情况
PLANT_OCCUPANCY = [[False for _ in range(PLANT_AREA_GRID[0])] for _ in range(PLANT_AREA_GRID[1])]

# 存储第九列上次截图，用于比较检测僵尸
PREV_ZOMBIE_CELLS = [None for _ in range(PLANT_AREA_GRID[1])]
ZOMBIE_DIFF_THRESHOLD = 20

# 记录向日葵生产线程和产出统计
SUNFLOWER_THREADS = {}

# 全局队列，存放僵尸风险行（1-indexed）
zombie_risk_queue = Queue()

# 全局字典记录仙人掌部署失败次数，key为行号（1-indexed）
CACTUS_FAIL_COUNT = {}

# 全局字典记录冰炮上次发射时间（行号：时间）
ICE_CANNON_LAST_FIRE = {}

# 全局变量保存每个格子的初始状态（以 numpy 数组存储），结构与 PLANT_OCCUPANCY 相同
INITIAL_CELL_STATE = [[None for _ in range(PLANT_AREA_GRID[0])] for _ in range(PLANT_AREA_GRID[1])]
CELL_DIFF_THRESHOLD = 15  # 阈值，可根据测试调整

def run_adb_command(cmd: list) -> str:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()

def get_current_sun_count() -> int:
    image = screenshot_obj.capture()
    # 裁剪阳光区域：[84,25,193,78]
    cropped = image.crop((84, 25, 193, 78))
    # 将crop后的PIL图像转换为numpy数组再传给easyOCR
    result = reader.readtext(np.array(cropped), detail=0, paragraph=True)
    text = " ".join(result)
    try:
        sun_count = int(''.join(filter(str.isdigit, text)))
    except Exception:
        sun_count = 0
    return sun_count

def get_grid_occupaied(row: int, col: int) -> bool:
    return PLANT_OCCUPANCY[row - 1][col - 1]

def get_grid_info(row: int, col: int) -> dict:
    # TODO: 接入 Yolo 后用于识别当前格子植物信息
    return {}

def get_plant_card_cost(index: int) -> int:
    # TODO: 接入 Yolo 后用于识别植物卡片费用
    return 0

def get_plant_card_cooldown(index: int) -> bool:
    # TODO: 接入 Yolo 后用于识别植物卡片冷却状态
    return 0.0

def try_plant_plant(row: int, col: int, plant_card_index: int) -> bool:
    print(f"尝试在({row}, {col})种植植物卡{plant_card_index}")
    global PLANT_OCCUPANCY
    if PLANT_OCCUPANCY[row - 1][col - 1]:
        print(f"格子({row}, {col})已有植物，跳过种植")
        return False

    # 根据传入的1-indexed行列计算对应格子中心坐标
    base_x, base_y = PLANT_AREA_TOP_LEFT
    area_width, area_height = PLANT_AREA_SIZE
    cols, rows = PLANT_AREA_GRID
    cell_w = area_width / cols
    cell_h = area_height / rows
    grid_center_x = int(base_x + (col - 1) * cell_w + cell_w / 2)
    grid_center_y = int(base_y + (row - 1) * cell_h + cell_h / 2)
    
    # 获取种植前的阳光数量
    initial_sun = get_current_sun_count()
    
    # 模拟点击植物卡片，根据debug_drag_plant_cards中的坐标
    card_first_center_x = 103
    card_first_center_y = 180
    card_height = 107
    card_center_y = int(card_first_center_y + (plant_card_index - 1) * card_height)
    cmd_tap_card = ["adb", "shell", "input", "tap", str(card_first_center_x), str(card_center_y)]
    run_adb_command(cmd_tap_card)
    print(f"点击植物卡{plant_card_index}，坐标：({card_first_center_x}, {card_center_y})")
    
    time.sleep(0.2)  # 等待植物卡响应
    
    # 模拟点击格子中心位置进行种植
    cmd_tap_grid = ["adb", "shell", "input", "tap", str(grid_center_x), str(grid_center_y)]
    run_adb_command(cmd_tap_grid)
    print(f"在格子({row}, {col})中心({grid_center_x}, {grid_center_y})点击种植")
    
    time.sleep(0.5)  # 等待操作生效
    
    # 获取种植后的阳光数量
    new_sun = get_current_sun_count()
    
    # 如果阳光数减少则认为种植成功，同时记录该格子已存在植物
    if new_sun < initial_sun:
        PLANT_OCCUPANCY[row - 1][col - 1] = True
        print("种植成功")
        return True
    else:
        print("种植失败")
        return False
    
def detect_touch_input() -> None:
    cols, rows = PLANT_AREA_GRID
    # 计算单元格宽度，用以确定滑动半径
    area_width, _ = PLANT_AREA_SIZE
    cell_w = area_width / cols
    radius = int(cell_w / 2)
    try:
        # 尝试读取触摸输入，等待1秒
        proc = subprocess.run(["cat", "/dev/input/event4"], stdout=subprocess.PIPE, timeout=1)
        if proc.stdout:
            print("检测到触摸输入")
        else:
            raise subprocess.TimeoutExpired("cat", 1)
    except subprocess.TimeoutExpired:
        # 超时后无触摸输入，遍历第一列所有行进行阳光收集
        for row in range(1, rows + 1):
            collect_sun_from_grid(row, 1, radius)
    except Exception as e:
        print(e)

def collect_sun_from_grid(row: int, col: int, radius: int) -> None:
    print(f"正在收集第{row}行第{col}列的阳光")
    # 根据传入的行列计算格子中心坐标
    base_x, base_y = PLANT_AREA_TOP_LEFT
    area_width, area_height = PLANT_AREA_SIZE
    total_cols, total_rows = PLANT_AREA_GRID
    cell_w = area_width / total_cols
    cell_h = area_height / total_rows
    center_x = int(base_x + (col - 1) * cell_w + cell_w / 2)
    center_y = int(base_y + (row - 1) * cell_h + cell_h / 2)

    swipe_duration = "300"
    cmd_swipe_up = ["adb", "shell", "input", "swipe", str(center_x), str(center_y), str(center_x), str(center_y - radius), swipe_duration]
    cmd_swipe_down = ["adb", "shell", "input", "swipe", str(center_x), str(center_y), str(center_x), str(center_y + radius), swipe_duration]
    cmd_swipe_left = ["adb", "shell", "input", "swipe", str(center_x), str(center_y), str(center_x - radius), str(center_y), swipe_duration]
    cmd_swipe_right = ["adb", "shell", "input", "swipe", str(center_x), str(center_y), str(center_x + radius), str(center_y), swipe_duration]

    for cmd in [cmd_swipe_up, cmd_swipe_down, cmd_swipe_left, cmd_swipe_right]:
        run_adb_command(cmd)

    print(f"在格子({row}, {col})中心({center_x}, {center_y})使用半径{radius}滑动收集阳光")

def has_touch_event() -> bool:
    # 非阻塞检测触摸输入，使用0.1秒超时
    try:
        proc = subprocess.run(["cat", "/dev/input/event4"], stdout=subprocess.PIPE, timeout=0.1)
        return bool(proc.stdout)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

def collect_sun_from_column(col: int) -> None:
    print(f"尝试从第{col}列收集阳光")
    # 遍历指定列的所有行，使用格子宽度的一半作为半径采集阳光
    area_width, _ = PLANT_AREA_SIZE
    total_cols, total_rows = PLANT_AREA_GRID
    radius = int((area_width / total_cols) / 2)
    for row in range(1, total_rows + 1):
        if has_touch_event():
            print("检测到触摸输入，停止采集")
            break
        collect_sun_from_grid(row, col, radius)
    print(f"第{col}列阳光收集完成")

def collect_sun_from_column_parallel(col: int) -> None:
    threads = []
    total_rows = PLANT_AREA_GRID[1]
    for row in range(1, total_rows + 1):
        t = threading.Thread(target=collect_sun_from_grid, args=(row, col, int(PLANT_AREA_SIZE[0] / PLANT_AREA_GRID[0] / 2)))
        t.daemon = True
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

def debug_drag_plant_cards() -> None:
    first_center_x = 103
    first_center_y = 180
    card_height = 107
    drag_offset = 50  # 向右平移50像素用于检测
    for i in range(8):
        # 计算每个卡片中心坐标，每个卡片纵向递增card_height
        center_x = first_center_x
        center_y = first_center_y + i * card_height
        target_x = center_x + drag_offset
        target_y = center_y
        # 发送adb swipe命令模拟拖动（持续300毫秒）
        cmd = ["adb", "shell", "input", "swipe",
               str(center_x), str(center_y),
               str(target_x), str(target_y), "300"]
        print(f"拖动植物卡{i+1}: {center_x},{center_y} -> {target_x},{target_y}")
        run_adb_command(cmd)

def debug_plant_card_costs() -> None:
    # 确保图片保存目录存在
    img_dir = r"./img"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # 截取整屏图片
    proc = subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=subprocess.PIPE)
    image_bytes = proc.stdout
    image = Image.open(io.BytesIO(image_bytes))
    # 根据已校正的卡片中心坐标和卡片高度
    first_center_x = 100
    first_center_y = 180
    card_height = 107
    for i in range(8):
        center_x = first_center_x
        center_y = first_center_y + i * card_height
        region = (center_x, center_y, center_x + 70, center_y + 46)
        cost_img = image.crop(region)
        # 保存原图区域
        cost_img.save(os.path.join(img_dir, f"plant_card{i+1}_org.png"))
        # 转换为灰度
        gray = cost_img.convert("L")
        gray_path = os.path.join(img_dir, f"plant_card{i+1}_gray.png")
        gray.save(gray_path)
        # 二值化预处理，使用"L"模式生成8位灰度图像（阈值可调整）
        thresh = gray.point(lambda x: 0 if x < 100 else 255, "L")
        thresh_path = os.path.join(img_dir, f"plant_card{i+1}_thresh.png")
        thresh.save(thresh_path)
        # 同时尝试两种图片的OCR识别
        text_list_gray = reader.readtext(np.array(gray), detail=0, paragraph=True)
        text_gray = " ".join(text_list_gray)
        text_list_thresh = reader.readtext(np.array(thresh), detail=0, paragraph=True)
        text_thresh = " ".join(text_list_thresh)
        cost_gray = ''.join(filter(str.isdigit, text_gray))
        cost_thresh = ''.join(filter(str.isdigit, text_thresh))
        print(f"植物卡{i+1}费用区域: {region}")
        print(f"  gray识别内容: '{text_gray}'，费用: {cost_gray}")
        print(f"  thresh识别内容: '{text_thresh}'，费用: {cost_thresh}")

def sunflower_production_thread(row: int, col: int):
    while True:
        time.sleep(25)  # 25秒生产周期
        collect_sun_from_grid(row, col, int(PLANT_AREA_SIZE[0] / PLANT_AREA_GRID[0] / 2))

def start_sunflower_production(row: int, col: int):
    """
    如果指定格子未启动向日葵生产，则新建一个后台线程
    """
    key = (row, col)
    if key not in SUNFLOWER_THREADS:
        t = threading.Thread(target=sunflower_production_thread, args=(row, col))
        t.daemon = True
        SUNFLOWER_THREADS[key] = t
        t.start()
        print(f"启动向日葵({row},{col})生产线程")

def plant(row: int, col: int, card: int) -> None:
    # 如果指定位置已有植物，则直接返回
    if get_grid_occupaied(row, col):
        print(f"({row},{col})已有植物，跳过种植")
        return
    while not get_grid_occupaied(row, col):
        success = try_plant_plant(row, col, card)
        if success and card == 7:
            start_sunflower_production(row, col)
            collect_sun_from_grid(row, col, int(PLANT_AREA_SIZE[0]/PLANT_AREA_GRID[0]/2))
        elif not success:
            # 种植失败时，尝试收集所有第一列已种植向日葵的阳光
            print(f"({row},{col})种植失败，尝试收集已种植向日葵的阳光")
            total_rows = PLANT_AREA_GRID[1]
            for r in range(1, total_rows+1):
                if get_grid_occupaied(r, 1):
                    collect_sun_from_grid(r, 1, int(PLANT_AREA_SIZE[0]/PLANT_AREA_GRID[0]/2))
            time.sleep(1)  # 等待一段时间后重试
        continue
    print(f"({row},{col})种植完成")

def detect_zombies_in_ninth_column() -> list:
    """
    截图第九列每个格子，并与上一次截图对比，
    返回检测到明显变化（疑似僵尸出现）的行（1-indexed）列表。
    """
    global PREV_ZOMBIE_CELLS
    zombie_rows = []
    # 使用统一截图器获取整屏截图
    full_image = screenshot_obj.capture()
    full_image = full_image.convert("L")  # 使用灰度图降低噪声
    
    # 计算第九列（最后一列）格子区域
    base_x, base_y = PLANT_AREA_TOP_LEFT
    area_width, area_height = PLANT_AREA_SIZE
    total_cols, total_rows = PLANT_AREA_GRID
    cell_w = area_width / total_cols
    cell_h = area_height / total_rows
    col = total_cols  # 第九列
    for row in range(1, total_rows + 1):
        left = int(base_x + (col - 1) * cell_w)
        upper = int(base_y + (row - 1) * cell_h)
        right = int(left + cell_w)
        lower = int(upper + cell_h)
        cell_img = full_image.crop((left, upper, right, lower))
        cell_array = np.array(cell_img, dtype=np.int16)
        # 对比前一次截图
        prev = PREV_ZOMBIE_CELLS[row - 1]
        if prev is not None:
            diff = np.mean(np.abs(cell_array - prev))
            if diff > ZOMBIE_DIFF_THRESHOLD:
                zombie_rows.append(row)
        # 更新当前图像
        PREV_ZOMBIE_CELLS[row - 1] = cell_array
    return zombie_rows

def zombie_detection_thread():
    """
    定时检测第九列僵尸风险，将检测到的风险行放入队列中
    """
    while True:
        zombies = detect_zombies_in_ninth_column()
        if zombies:
            for r in zombies:
                zombie_risk_queue.put(r)
        time.sleep(1)  # 每秒检查一次

def fire_ice_cannon(row: int) -> bool:
    """
    对指定行的冰炮（假设部署在第三列）进行发射：
    – 判断若上次发射未超过15秒，返回False；
    – 计算冰炮坐标以及目标格子（取该行第八列，即僵尸前一格）的坐标，
      模拟点击依次选择冰炮和目标；
    – 更新发射时间并认为发射成功(可后续加入检测逻辑)。
    """
    current_time = time.time()
    # 判断冷却时间：15 s
    last_fire = ICE_CANNON_LAST_FIRE.get(row, 0)
    if current_time - last_fire < 15:
        print(f"行{row}冰炮仍在冷却({current_time - last_fire:.1f}s)，不可发射")
        return False

    # 计算冰炮所在的格子坐标（第三列）；
    base_x, base_y = PLANT_AREA_TOP_LEFT
    area_width, area_height = PLANT_AREA_SIZE
    total_cols, total_rows = PLANT_AREA_GRID
    cell_w = area_width / total_cols
    cell_h = area_height / total_rows
    # 冰炮部署在第三列
    cannon_x = int(base_x + (3 - 1) * cell_w + cell_w / 2)
    cannon_y = int(base_y + (row - 1) * cell_h + cell_h / 2)
    # 预判目标为该行第8列（假设僵尸在9列，冰炮发射到前一格）
    target_x = int(base_x + (8 - 1) * cell_w + cell_w / 2)
    target_y = int(base_y + (row - 1) * cell_h + cell_h / 2)
    print(f"行{row}冰炮发射：从({cannon_x}, {cannon_y}) -> ({target_x}, {target_y})")
    # 模拟点击冰炮（选择）和目标
    cmd_fire1 = ["adb", "shell", "input", "tap", str(cannon_x), str(cannon_y)]
    cmd_fire2 = ["adb", "shell", "input", "tap", str(target_x), str(target_y)]
    run_adb_command(cmd_fire1)
    time.sleep(0.2)
    run_adb_command(cmd_fire2)
    print(f"行{row}冰炮发射指令已发出")
    # 更新发射时间
    ICE_CANNON_LAST_FIRE[row] = time.time()
    return True

def init_initial_cell_state():
    """程序开始时获取每个格子的初始状态"""
    global INITIAL_CELL_STATE
    full_image = screenshot_obj.capture()
    base_x, base_y = PLANT_AREA_TOP_LEFT
    area_width, area_height = PLANT_AREA_SIZE
    total_cols, total_rows = PLANT_AREA_GRID
    cell_w = area_width / total_cols
    cell_h = area_height / total_rows
    for r in range(total_rows):
        for c in range(total_cols):
            left = int(base_x + c * cell_w)
            upper = int(base_y + r * cell_h)
            right = int(left + cell_w)
            lower = int(upper + cell_h)
            cell_img = full_image.crop((left, upper, right, lower)).convert("L")
            INITIAL_CELL_STATE[r][c] = np.array(cell_img, dtype=np.int16)

def cell_has_zombie(row: int, col: int) -> bool:
    """
    比较当前格子与初始状态的差异，若均值绝对差超过阈值则认为该格子上有僵尸。
    行、列均为1-indexed。
    """
    base_x, base_y = PLANT_AREA_TOP_LEFT
    area_width, area_height = PLANT_AREA_SIZE
    total_cols, total_rows = PLANT_AREA_GRID
    cell_w = area_width / total_cols
    cell_h = area_height / total_rows
    left = int(base_x + (col - 1) * cell_w)
    upper = int(base_y + (row - 1) * cell_h)
    right = int(left + cell_w)
    lower = int(upper + cell_h)
    current_img = screenshot_obj.capture().crop((left, upper, right, lower)).convert("L")
    current_array = np.array(current_img, dtype=np.int16)
    initial_array = INITIAL_CELL_STATE[row - 1][col - 1]
    diff = np.mean(np.abs(current_array - initial_array))
    return diff > CELL_DIFF_THRESHOLD

def get_zombie_column(row: int) -> int:
    """
    对指定行（1-indexed），从最右侧开始依次检测各列，
    返回检测到的第一个不同于初始状态的列号（1-indexed），若都相同返回 None。
    """
    total_cols = PLANT_AREA_GRID[0]
    for col in range(total_cols, 0, -1):
        if cell_has_zombie(row, col):
            return col
    return None

def fire_ice_cannon(row: int) -> bool:
    """
    对指定行的冰炮（部署在第三列）进行发射：
    – 判断若上次发射未超过15秒返回False；
    – 根据 get_zombie_column(row) 动态获取僵尸所在列，目标为僵尸前一格（至少为1）；
    – 模拟点击冰炮和目标，更新发射时间。
    """
    current_time = time.time()
    last_fire = ICE_CANNON_LAST_FIRE.get(row, 0)
    if current_time - last_fire < 15:
        print(f"行{row}冰炮仍在冷却({current_time - last_fire:.1f}s)，不可发射")
        return False
    zombie_col = get_zombie_column(row)
    if zombie_col is None:
        print(f"行{row}检测不到僵尸，不发射冰炮")
        return False
    # 目标为僵尸前一格，确保不小于1
    target_col = max(zombie_col - 1, 1)
    base_x, base_y = PLANT_AREA_TOP_LEFT
    area_width, area_height = PLANT_AREA_SIZE
    total_cols, total_rows = PLANT_AREA_GRID
    cell_w = area_width / total_cols
    cell_h = area_height / total_rows
    # 冰炮位于第三列
    cannon_x = int(base_x + (3 - 1) * cell_w + cell_w / 2)
    cannon_y = int(base_y + (row - 1) * cell_h + cell_h / 2)
    target_x = int(base_x + (target_col - 1) * cell_w + cell_w / 2)
    target_y = cannon_y  # 同一行
    print(f"行{row}冰炮发射：从({cannon_x},{cannon_y}) -> ({target_x},{target_y}) [目标根据僵尸列{zombie_col}]")
    run_adb_command(["adb", "shell", "input", "tap", str(cannon_x), str(cannon_y)])
    time.sleep(0.2)
    run_adb_command(["adb", "shell", "input", "tap", str(target_x), str(target_y)])
    ICE_CANNON_LAST_FIRE[row] = time.time()
    return True

# 启动僵尸检测线程
zombie_detector = threading.Thread(target=zombie_detection_thread, args=())
zombie_detector.daemon = True
zombie_detector.start()

# 在程序开始前调用初始状态采集
init_initial_cell_state()

# 修改主函数部署策略
if __name__ == "__main__":
    # 基础阵容
    while True:
        # 优先处理僵尸风险队列中的行
        while not zombie_risk_queue.empty():
            risk_row = zombie_risk_queue.get()
            # 若该行第二列未部署植物，则尝试部署仙人掌
            if not get_grid_occupaied(risk_row, 2):
                print(f"检测到行{risk_row}有僵尸风险，部署仙人掌防守")
                if not try_plant_plant(risk_row, 2, 2):
                    # 如果失败，增加该行的失败计数
                    CACTUS_FAIL_COUNT[risk_row] = CACTUS_FAIL_COUNT.get(risk_row, 0) + 1
                    if CACTUS_FAIL_COUNT[risk_row] > 2:
                        print(f"行{risk_row}仙人掌部署连续失败，尝试在第一列部署向日葵以增加阳光")
                        try_plant_plant(risk_row, 1, 7)
                else:
                    # 成功后重置失败计数
                    CACTUS_FAIL_COUNT[risk_row] = 0
            # 如果该行第三列已部署冰炮，则立即尝试发射冰炮
            if get_grid_occupaied(risk_row, 3):
                fire_ice_cannon(risk_row)
        # 若无僵尸风险，则在第一列部署向日葵（若未部署）
        total_rows = PLANT_AREA_GRID[1]
        for row in range(1, total_rows + 1):
            if not get_grid_occupaied(row, 1):
                print(f"无僵尸风险，部署向日葵于第一列的第{row}行")
            if try_plant_plant(row, 1, 7):
                collect_sun_from_grid(row, 1, int(PLANT_AREA_SIZE[0] / PLANT_AREA_GRID[0] / 2))
            if zombie_risk_queue.empty()==False:
                break
            # 同时，对每行第三列（冰炮）若已部署，可实时尝试发射冰炮
            if get_grid_occupaied(row, 3):
                fire_ice_cannon(row)
        # 使用并行收集方式采集第一列阳光
        collect_sun_from_column_parallel(1)
        time.sleep(1)

        if all(PLANT_OCCUPANCY[r][0] and PLANT_OCCUPANCY[r][1] for r in range(len(PLANT_OCCUPANCY))):
            break
    
    # 第三列
    while True:
        collect_sun_from_column_parallel(1)
        time.sleep(1)

        # 部署冰炮整列（第三列，每行各一个）
        total_rows = PLANT_AREA_GRID[1]
        for row in range(1, total_rows + 1):
            if not get_grid_occupaied(row, 3):
                print(f"部署冰炮于({row},3)")
                while not get_grid_occupaied(row, 3):
                    try_plant_plant(row, 3, 8)
                    time.sleep(1)
                    if not zombie_risk_queue.empty():
                        break
            if zombie_risk_queue.empty()==False:
                break

        # 监测僵尸风险，触发相应行的冰炮发射
        print("开始监测并使用冰炮打击僵尸")
        while True:
            if not zombie_risk_queue.empty():
                risk_row = zombie_risk_queue.get()
                print(f"检测到行{risk_row}存在僵尸风险，尝试发射冰炮")
                if not fire_ice_cannon(risk_row):
                    print(f"行{risk_row}冰炮冷却中，发射失败")
            time.sleep(0.5)