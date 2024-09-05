import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import csv

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
    csv_writer.writerow(['Filename', 'Grayscale Area (pixels)', 'Green Mask Area (pixels)'])

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
    _, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # HSV画像に変換して緑色抽出
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([0, 6, 0])
    upper_green = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # ノイズ除去処理（モルフォロジー）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 輪郭検出（グレースケール）
    contours_gray, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours_gray = image.copy()
    cv2.drawContours(image_with_contours_gray, contours_gray, -1, (0, 255, 0), 2)

    # 輪郭検出（緑抽出）
    contours_green, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours_green = image.copy()
    cv2.drawContours(image_with_contours_green, contours_green, -1, (0, 255, 0), 2)

    # 面積計算（グレースケール）
    area_gray = np.sum(threshold_image == 255)

    # 面積計算（緑抽出）
    area_green = np.sum(mask == 255)

    # グレースケール境界画像に面積を表示
    cv2.putText(image_with_contours_gray, f'Area: {area_gray} pixels', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 緑のマスク境界画像に面積を表示
    cv2.putText(image_with_contours_green, f'Area: {area_green} pixels', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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
    plt.imshow(cv2.cvtColor(image_with_contours_gray, cv2.COLOR_BGR2RGB))
    plt.title(f'Grayscale Contours\nArea: {area_gray} pixels')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(image_with_contours_green, cv2.COLOR_BGR2RGB))
    plt.title(f'Green Contours\nArea: {area_green} pixels')

    plt.tight_layout()

    # 書き出すために結果を保存
    plt.savefig(result_image_path)
    plt.close()  # メモリ節約のためウィンドウを閉じる

    # 画像ごとの結果をCSVに追記
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([image_file, area_gray, area_green])

    print(f"Processed and saved: {result_image_path}")