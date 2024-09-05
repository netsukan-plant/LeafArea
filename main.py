import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import csv
from proofreading import calculate_pixel_per_cm

# 閾値ファイルを読み込む
thresholds_file = 'thresholds.csv'

# 閾値の読み込み関数
def load_thresholds(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        thresholds = next(csv_reader)  # 最初の行を取得
        return {
            'grayscale_threshold': int(thresholds['grayscale_threshold']),
            'lower_hue': int(thresholds['lower_hue']),
            'lower_saturation': int(thresholds['lower_saturation']),
            'lower_value': int(thresholds['lower_value']),
            'upper_hue': int(thresholds['upper_hue']),
            'upper_saturation': int(thresholds['upper_saturation']),
            'upper_value': int(thresholds['upper_value']),
        }

# 閾値をロード
thresholds = load_thresholds(thresholds_file)

# 2cm×2cmの黒い四角形が含まれる画像のパス
reference_image_path = 'reference_square.jpg'
pixel_per_cm = calculate_pixel_per_cm(reference_image_path)

# 入力画像フォルダと出力結果フォルダのパス
input_folder = 'images'
output_folder = 'results'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 実行時刻を取得して、フォルダ名に使用
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 実行時刻を基にしたフォルダパスを作成
output_folder_with_time = os.path.join(output_folder, current_time)

# 実行時刻を基にしたフォルダパスを作成
output_folder_with_time_image = os.path.join(output_folder, current_time, 'image')

# 実行時刻で名前を付けたフォルダを作成
if not os.path.exists(output_folder_with_time):
    os.makedirs(output_folder_with_time)
if not os.path.exists(output_folder_with_time_image):
    os.makedirs(output_folder_with_time_image)

# CSVファイルのパスを指定
csv_file_path = os.path.join(output_folder_with_time, 'results.csv')

# CSVファイルを作成し、ヘッダーを書き込む
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Filename', 'Grayscale Area (cm^2)', 'Green Mask Area (cm^2)'])

# 画像フォルダからすべての画像ファイルを取得
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # 画像を読み込む
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # 画像が読み込まれていない場合、スキップ
    if image is None:
        print(f"Error: Image '{image_file}' not found or cannot be opened.")
        continue

    # 拡張子を除いたファイル名部分だけを取得
    file_name_without_ext = os.path.splitext(image_file)[0]

    # グレースケール画像に変換
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # グレースケールの閾値処理
    _, threshold_image = cv2.threshold(gray_image, thresholds['grayscale_threshold'], 255, cv2.THRESH_BINARY)

    # HSV画像に変換して緑色抽出
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([thresholds['lower_hue'], thresholds['lower_saturation'], thresholds['lower_value']])
    upper_green = np.array([thresholds['upper_hue'], thresholds['upper_saturation'], thresholds['upper_value']])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # ノイズ除去処理（モルフォロジー）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 面積計算（グレースケール）
    area_gray_pixels = np.sum(threshold_image == 255)
    area_gray_cm2 = area_gray_pixels / (pixel_per_cm ** 2)  # ピクセル数から平方センチメートルに変換

    # 面積計算（緑抽出）
    area_green_pixels = np.sum(mask == 255)
    area_green_cm2 = area_green_pixels / (pixel_per_cm ** 2)  # ピクセル数から平方センチメートルに変換

    # グレースケール境界画像に面積を表示
    cv2.putText(image, f'Grayscale Area: {area_gray_cm2:.2f} cm^2', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 緑のマスク境界画像に面積を表示
    cv2.putText(image, f'Green Area: {area_green_cm2:.2f} cm^2', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 結果の画像を保存
    result_image_path = os.path.join(output_folder_with_time_image, f'result_{file_name_without_ext}.jpg')
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(threshold_image, cmap='gray')
    plt.title('Grayscale Threshold Image')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Grayscale Contours\nArea: {area_gray_cm2:.2f} cm^2')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Green Contours\nArea: {area_green_cm2:.2f} cm^2')

    plt.tight_layout()

    # 書き出すために結果を保存
    plt.savefig(result_image_path)
    plt.close()  # メモリ節約のためウィンドウを閉じる

    # 画像ごとの結果をCSVに追記
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([image_file, area_gray_cm2, area_green_cm2])

    print(f"Processed and saved: {result_image_path}")