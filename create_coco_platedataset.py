import os
import cv2
import numpy as np
import random
import json
from PIL import Image, ImageDraw, ImageFont
import argparse


TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TEST_DIR = 'test/'

BASE_IMAGE_FOLDER = "./background_images"

# ---------------------------
# ここにある create_* 系の関数や
# generate_plate_colors() などは
# 元コードから流用してください
# ---------------------------
# ラベル名を統合して一意なインデックスを割り当て
def create_all_labels():
    hiragana = create_hiragana()
    hiragana_alphabet = ['Y', 'B', 'E', 'H', 'K', 'M']
    digits = [str(i) for i in range(10)]
    place = create_place_list()

    all_labels = digits + hiragana + hiragana_alphabet + place
    return all_labels

def create_all_labels_roma():
    hiragana = create_hiragana_roma()
    hiragana_alphabet = ['Y', 'B', 'E', 'H', 'K', 'M']
    digits = [str(i) for i in range(10)]
    place = create_place_list_roma()

    all_labels = digits + hiragana + hiragana_alphabet + place
    return all_labels
# 地名リストを作成
def create_place_list():
    return [
        "札幌", "函館", "旭川", "室蘭", "苫小牧", "釧路", "知床", "帯広", "北見", "小樽", "千歳",
        "青森", "弘前", "八戸", "岩手", "盛岡", "平泉", "宮城", "仙台", "秋田", "山形", "庄内", "福島", 
        "会津", "郡山", "白河", "いわき", "水戸", "日立", "土浦", "つくば", "栃木", "那須", "とちぎ", "宇都宮", 
        "足利", "群馬", "前橋", "高崎", "伊勢崎", "桐生", "埼玉", "川口", "所沢", "川越", "熊谷", "春日部", 
        "越谷", "千葉", "成田", "習志野", "市川", "船橋", "袖ヶ浦", "市原", "野田", "柏", "松戸", "品川", 
        "世田谷", "練馬", "杉並", "板橋", "足立", "江東", "葛飾", "八王子", "多摩", "神奈川", "横浜", 
        "川崎", "湘南", "相模", "相模原", "山梨", "富士山", "新潟", "長岡", "上越", "長野", "松本", "諏訪", 
        "富山", "石川", "金沢", "福井", "岐阜", "飛騨", "静岡", "浜松", "沼津", "伊豆", "富士", "御殿場", 
        "愛知", "豊橋", "三河", "岡崎", "豊田", "尾張小牧", "一宮", "春日井", "名古屋", "三重", "鈴鹿", 
        "四日市", "伊勢志摩", "滋賀", "京都", "大阪", "和泉","堺", "吹田", "高槻", "奈良", "飛鳥", 
        "和歌山", "兵庫", "姫路", "鳥取", "島根", "出雲", "岡山", "倉敷", "津山", "広島", "福山", 
        "呉", "三次", "山口", "下関", "徳島", "香川", "高松", "丸亀", "愛媛", "高知", "福岡", "北九州", 
        "久留米", "筑豊", "博多", "宗像", "佐賀", "長崎", "佐世保", "長崎市", "熊本", "大分", "宮崎", 
        "鹿児島", "奄美", "沖縄", "なにわ","神戸","大宮"
    ]

def create_place_list_roma():
    return [
        "sapporo", "hakodate", "asahikawa", "muroran", "tomakomai", "kushiro", "shiretoko", "obihiro", "kitami", "otaru", "chitose",
        "aomori", "hirosaki", "hachinohe", "iwate", "morioka", "hiraizumi", "miyagi", "sendai", "akita", "yamagata", "shonai", 
        "fukushima", "aizu", "koriyama", "shirakawa", "iwaki", "mito", "hitachi", "tsuchiura", "tsukuba", "tochigi", "nasu", 
        "hiragana_tochigi", "utsunomiya", "ashikaga", "gunma", "maebashi", "takasaki", "saitama","isesaki","kityu", "kawaguchi", "tokorozawa", "kawagoe", 
        "kumagaya", "kasukabe", "koshigaya", "chiba", "narita", "narashino", "ichikawa", "funabashi", "sodegaura", "ichihara", 
        "noda", "kashiwa", "matsudo", "shinagawa", "setagaya", "nerima", "suginami", "itabashi", "adachi", "koto", "katsushika", 
        "hachioji", "tama", "kanagawa", "yokohama", "kawasaki", "shonan", "sagami","sagamihara", "yamanashi", "fujisan", "niigata", "nagaoka", 
        "joetsu", "nagano", "matsumoto", "suwa", "toyama", "ishikawa", "kanazawa", "fukui", "gifu", "hida", "shizuoka", "hamamatsu", 
        "numazu", "izu","fuji","gotenba", "aichi", "toyohashi", "mikawa", "okazaki", "toyota", "owarikomaki", "ichinomiya", "kasugai", "nagoya","mie", 
        "suzuka", "yokkaichi", "iseshima", "shiga", "kyoto", "osaka", "izumi", "sakai","suita","takatsuki", "nara", "asuka", "wakayama", "hyogo", 
        "himeji", "tottori", "shimane", "izumo", "okayama", "kurashiki","tsuyama", "hiroshima", "fukuyama","kure","miyoshi", "yamaguchi", "shimonoseki", 
        "tokushima", "kagawa", "takamatsu","marugame", "ehime", "kochi", "fukuoka", "kitakyushu", "kurume", "chikuho","hakata","munakata", "saga", "nagasaki", 
        "sasebo", "nagasakishi", "kumamoto", "oita", "miyazaki", "kagoshima", "amami", "okinawa","naniwa","kobe","omiya"
    ]

# ひらがなリストを作成
def create_hiragana():
    hiragana = []
    for i in range(ord('あ'), ord('お') + 1):
        if i % 2 == 0:
            hiragana.append(chr(i))
    for i in range(ord('か'), ord('ち') + 1):
        if i % 2 == 1:
            hiragana.append(chr(i))
    hiragana.append('つ')
    for i in range(ord('て'), ord('と') + 1):
        if i % 2 == 0:
            hiragana.append(chr(i))
    for i in range(ord('な'), ord('の') + 1):
        hiragana.append(chr(i))
    for i in range(ord('は'), ord('ほ') + 1):
        if i % 3 == 0:
            hiragana.append(chr(i))
    for i in range(ord('ま'), ord('も') + 1):
        hiragana.append(chr(i))
    for i in range(ord('や'), ord('よ') + 1):
        if i % 2 == 0:
            hiragana.append(chr(i))
    for i in range(ord('ら'), ord('ろ') + 1):
        hiragana.append(chr(i))
    hiragana.append('わ')
    hiragana.append('を')

    hiragana.remove('お')
    hiragana.remove('し')
    hiragana.remove('へ')

    return hiragana

def create_hiragana_roma():
    return  ['a', 'i', 'u', 'e', 'ka', 'ki', 'ku', 'ke', 'ko', 'sa', 'su', 'se', 'so', 'ta', 'chi', 'tsu', 'te', 'to', 'na', 'ni', 'nu', 'ne', 'no', 'ha', 'hi', 'fu', 'ho', 'ma', 'mi', 'mu', 'me', 'mo', 'ya', 'yu', 'yo', 'ra', 'ri', 'ru', 're', 'ro', 'wa', 'wo']


# ラベルインデックス取得
def get_label_index(label, all_labels):
    return all_labels.index(label)


# ラベルインデックス取得
def get_label_index(label, all_labels):
    return all_labels.index(label)

# ナンバープレートの色のランダム選択
def generate_plate_colors():
    color_int = random.randint(0, 99)
    if color_int < 60:
        plate_color = (255, 255, 255)
        text_color = (29, 54, 35)
    elif color_int < 80:
        plate_color = (230, 180, 21)
        text_color = (0, 0, 0)
    elif color_int < 95:
        plate_color = (29, 54, 35)
        text_color = (255, 255, 255)
    else:
        plate_color = (0, 0, 0)
        text_color = (230, 180, 21)
    return plate_color, text_color

def random_sample_file():
    """
    1～100の乱数をとり、60%:train, 20%:val, 20%:test の振り分けに
    """
    dir_rand = random.randint(1, 100)
    if dir_rand <= 70:
        return TRAIN_DIR
    elif dir_rand <= 90:
        return VAL_DIR
    else:
        return TEST_DIR

# 数字部分とアルファベットをランダムに生成
def generate_number_plate():
    first_digit = random.randint(0, 9)  # 最初の桁（数字）
    second_digit = random.choice([random.randint(0, 9), random.choice(['Y', 'B', 'E', 'H', 'K', 'M'])])  # 次の桁（数字またはアルファベット）
    third_digit = random.choice([random.randint(0, 9), random.choice(['Y', 'B', 'E', 'H', 'K', 'M'])])  # 最後の桁（数字またはアルファベット）
    return [first_digit, second_digit, third_digit]

#------------------------------

BASE_IMAGE_FOLDER = "./background_images"

def overlay_plate_on_background(plate_img, plate_corners):
    """
    背景画像にナンバープレート画像をランダム位置に合成。
    plate_img: プレート画像 (H, W, 3)
    plate_corners: プレートの4隅の座標 (4, 2)

    戻り値: 合成後画像, プレートの新しい4隅座標
    """
    # 背景画像をランダムに選択
    bg_files = [os.path.join(BASE_IMAGE_FOLDER, f) for f in os.listdir(BASE_IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
    bg_path = random.choice(bg_files)
    bg = cv2.imread(bg_path)

    bg_h, bg_w = bg.shape[:2]
    plate_h, plate_w = plate_img.shape[:2]

    # 貼り付け位置をランダムに決定（境界チェック）
    max_x = bg_w - plate_w
    max_y = bg_h - plate_h
    if max_x <= 0 or max_y <= 0:
        raise ValueError("背景画像が小さすぎます。プレートより大きな画像を用意してください。")

    offset_x = random.randint(0, max_x)
    offset_y = random.randint(0, max_y)

    # 合成処理
    roi = bg[offset_y:offset_y+plate_h, offset_x:offset_x+plate_w]
    mask = np.ones_like(plate_img) * 255  # 単純に白マスク（改善可）
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(plate_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg_roi = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg_roi = cv2.bitwise_and(plate_img, plate_img, mask=mask)
    dst = cv2.add(bg_roi, fg_roi)

    bg[offset_y:offset_y+plate_h, offset_x:offset_x+plate_w] = dst

    # 座標を更新（offset分ずらす）
    new_corners = plate_corners + np.array([offset_x, offset_y])

    return bg, new_corners

def is_collapsed(corners, threshold=1.0):
    """
    corners: ndarray (4, 2)
    threshold: float, stdのしきい値。小さすぎると点や直線になる。
    """
    x, y = corners[:, 0], corners[:, 1]
    return np.std(x) < threshold or np.std(y) < threshold


def polygon_area(points):
    """
    シンプルな多角形面積(2D)を求める関数
    points: [x0,y0, x1,y1, x2,y2, x3,y3,...]
    (Shoelace formula)
    """
    xys = np.array(points).reshape(-1,2)
    x = xys[:,0]
    y = xys[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

def generate_plate_corners_image(image_num_per_place, out_dir="coco_plate_dadtasets", verbose=False):
    """
    ナンバープレート画像を生成し、
    プレート全体の4隅をポリゴンとして記録し、
    COCO形式のJSONにまとめるサンプル。
    """
    os.makedirs(out_dir, exist_ok=True)
    # カテゴリは1つ (plate)
    category_id = 1
    category_list = [{"id": category_id, "name": "plate"}]

    # 辞書を分けて保持 (train, val, test)
    coco_data = {
        "train": {
            "images": [],
            "annotations": [],
            "categories": category_list
        },
        "val": {
            "images": [],
            "annotations": [],
            "categories": category_list
        },
        "test": {
            "images": [],
            "annotations": [],
            "categories": category_list
        }
    }

    # 画像ID,アノテーションIDを各split毎に管理
    image_id_dict = {"train": 1, "val": 1, "test": 1}
    ann_id_dict   = {"train": 1, "val": 1, "test": 1}


    # 生成枚数(image_num_per_place)の例
    # 本来は placeリストなどを使って繰り返しつくるのを短縮
    for idx in range(image_num_per_place):
        # (例) 330x165 のプレートを作成
        plate_width = 330
        plate_height = 165
        plate_color = (255, 255, 255)
        text_color = (0, 0, 0)

        str_place = random.choice(create_place_list())

        # PILでプレート画像作成
        im = Image.new("RGB", (plate_width, plate_height), plate_color)
        draw = ImageDraw.Draw(im)
        # 文字などを描画するならここで write text
        # 例: draw.text((...), "ABCD", fill=text_color, ...)
        # ランダムでナンバープレートの数字部分を生成
        num4 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(3)]  # 4桁の数字生成
        num3 = generate_number_plate()  # 3桁数字部分
        hiragana_selected = random.choice(create_hiragana())  # ひらがな部分
        place_selected = str_place  # 地名部分

        # ナンバープレートの背景色とテキスト色を決定
        plate_color, text_color = generate_plate_colors()

        # 画像作成
        width = 330
        height = 165
        im = Image.new("RGB", (width, height), plate_color)
        draw = ImageDraw.Draw(im)

        # フォントのランダム選択
        font_4num = ImageFont.truetype('./fonts/FZcarnumberJA-OTF_ver10.otf', 80)

        font_num3 = ImageFont.truetype('./fonts/FZcarnumberJA-OTF_ver10.otf', 40)

        # 地名のフォントをランダムに選択    
        font_random = random.randint(0, 2)
        if font_random == 0:
            font_place = ImageFont.truetype('./fonts/meiryo.ttc', 45)
        elif font_random == 1:
            font_place = ImageFont.truetype('./fonts/HGRSMP.TTF', 45)
        else:
            font_place = ImageFont.truetype('./fonts/HGRGM.TTC', 45)

        # メイリオでのひらがなフォント
        font_hira = ImageFont.truetype('./fonts/HGRSKP.TTF', 40)

        # 文字描画の初期位置（画像左上からx, yだけ離れた位置）
        x_4num = 70
        y_4num = 70
        if len(str_place) == 2:
            x_place = 84
            y_place = 10
        else:
            x_place = 74
            y_place = 12
        x_3num = 180
        y_3num = 2
        x_hira = 16
        y_hira = 90

        # 地名の描画位置調整（地名が2文字の場合、描画位置を調整）
        if len(place_selected) == 2:
            x_place = 84  # 2文字の地名の描画位置
            y_place = 10
        else:
            x_place = 74  # その他の地名の描画位置
            y_place = 12


        # 4桁の数字の描画とバウンディングボックス取得
        for i, text in enumerate(num4):
            text_str = str(text)
            draw.text((x_4num, y_4num), text_str, fill=text_color, font=font_4num)
            x_4num = x_4num + 55
            if i == 1:
                x_4num = x_4num + 33

        # 地名の描画とバウンディングボックス取得
        draw.text((x_place, y_place), place_selected, fill=text_color, font=font_place)
        bbox_place = draw.textbbox((x_place, y_place), place_selected, font=font_place)

        # bbox[2] は地名の右端のx座標
        x_3num = bbox_place[2] + 10  # 地名の右端から10px右にオフセット
        y_3num = y_place - 5   # 地名より少し上に配置（任意調整）

        # 3桁の描画とバウンディングボックス取得
        for i, text in enumerate(num3):
            text_str = str(text)
            draw.text((x_3num, y_3num), text_str, fill=text_color, font=font_num3)
            x_3num += 30  # 次の文字との間隔


        # ひらがなの描画とバウンディングボックス取得
        draw.text((x_hira, y_hira), hiragana_selected, fill= text_color, font=font_hira)
        bbox = draw.textbbox((x_hira, y_hira), hiragana_selected, font=font_hira)

        # ハイフンの描画（ラベルは作成しない）
        draw.line((185, 115, 205, 115), fill= text_color, width=10)

        # ネジ部分の描画
        draw.ellipse((50, 23, 60, 33), fill=(90, 80, 60))
        draw.ellipse((280, 23, 290, 33), fill=(90, 80, 60))


        # NumPyに変換
        cv_img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        # ----- 透視変換で軽度に歪ませる (例)-----
        transform_prob = random.random()
        rows, cols = cv_img.shape[:2]
        corners_src = np.array([
            [0, 0],
            [cols - 1, 0],
            [cols - 1, rows - 1],
            [0, rows - 1]
        ], dtype=np.float32)

        # 50%の確率で変形
        if transform_prob < 0.7:
            if verbose:
                print(f"[{idx}] Perspective transform applied (prob={transform_prob:.2f})")
            max_shift_x = cols * 0.1
            max_shift_y = rows * 0.1
            shift = np.column_stack([
                np.random.uniform(-max_shift_x, max_shift_x, size=4),
                np.random.uniform(-max_shift_y, max_shift_y, size=4)
            ]).astype(np.float32)
            corners_dst = corners_src + shift

            if is_collapsed(corners_dst):
                continue

            M = cv2.getPerspectiveTransform(corners_src, corners_dst)
            warped = cv2.warpPerspective(cv_img, M, (cols, rows))
            final_img = warped
            corners_transformed = cv2.perspectiveTransform(corners_src.reshape(-1, 1, 2), M).reshape(-1, 2)

            if is_collapsed(corners_transformed):
                continue
        else:
            if verbose:
                print(f"[{idx}] No transform (prob={transform_prob:.2f})")
            final_img = cv_img
            corners_transformed = corners_src

        final_img, corners_transformed = overlay_plate_on_background(final_img, corners_transformed)

        
        # ここで、正しい画像サイズを取得し直す
        height, width = final_img.shape[:2]
        corners_transformed = np.clip(corners_transformed, [0, 0], [width - 1, height - 1])

        corners_transformed = np.clip(corners_transformed, [0, 0], [width - 1, height - 1])

        if verbose:
            print("Updated corners:", corners_transformed)

        # corners_transformed = [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        # ここをCOCO形式の polygon として保存

        # 3) 出力先splitを決める
        split_dir = random_sample_file().strip('/')  # => "train"/"val"/"test"

        # 4) 画像保存先フォルダ
        images_dir = os.path.join(out_dir, split_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # 画像ファイル名
        file_name = f"plate_{idx}.jpg"
        out_path = os.path.join(images_dir, file_name)
        cv2.imwrite(out_path, final_img)

        # 画像情報
        height, width = final_img.shape[:2]
        # COCO images[]
        img_info = {
            "id": image_id_dict[split_dir],
            "file_name": file_name,
            "height": height,
            "width": width
        }
        coco_data[split_dir]["images"].append(img_info)

        # 5) 四隅ポリゴン例 (軸平行の場合)
        corners_src = np.float32([
            [0,0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ])
        # flatten
        poly = corners_transformed.flatten().tolist()  # [x0,y0,x1,y1,x2,y2,x3,y3]

        # bbox
        x_vals = corners_transformed[:,0]
        y_vals = corners_transformed[:,1]
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        w_box = x_max - x_min
        h_box = y_max - y_min
        area_val = polygon_area(poly)

        # COCO annotations[]
        ann_info = {
            "id": ann_id_dict[split_dir],
            "image_id": image_id_dict[split_dir],
            "category_id": category_id,
            "iscrowd": 0,
            "segmentation": [poly],
            "bbox": [x_min, y_min, w_box, h_box],
            "area": area_val
        }
        coco_data[split_dir]["annotations"].append(ann_info)

        # IDカウントアップ
        image_id_dict[split_dir] += 1
        ann_id_dict[split_dir] += 1

    # 6) split毎に JSON 出力
    for split in ["train", "val", "test"]:
        final_dict = {
            "images": coco_data[split]["images"],
            "annotations": coco_data[split]["annotations"],
            "categories": coco_data[split]["categories"]
        }
        json_path = os.path.join(out_dir, f"{split}.json")
        with open(json_path, "w") as f:
            json.dump(final_dict, f, indent=2)
        print(f"Saved COCO {split}.json => {json_path}")


def polygon_area(points):
    """
    シンプルな多角形面積(2D)を求める関数
    points: [x0,y0, x1,y1, x2,y2, ..., x_{n-1},y_{n-1}]
    参考: シュー(靴)の公式
    """
    xys = np.array(points).reshape(-1,2)
    #  https://en.wikipedia.org/wiki/Shoelace_formula
    x = xys[:,0]
    y = xys[:,1]
    return 0.5 * np.abs( np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)) )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_num_per_place', type=int, default=5)
    parser.add_argument('--training_name', type=str, default="dataset_polygon")
    parser.add_argument('--verbose', type=bool,default=False )
    args = parser.parse_args()

    generate_plate_corners_image(args.image_num_per_place, args.training_name, args.verbose)
