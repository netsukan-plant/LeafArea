import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import csv
from proofreading import calculate_pixel_per_cm

# モード選択のための入力
def select_mode():
    print("モードを選択してください:")
    print("1: ノイズキャンセルを行う")
    print("2: ノイズキャンセルを行わない")
    
    while True:
        choice = input("モード番号を入力してください (1 または 2): ")
        if choice == '1':
            print("ノイズキャンセルを実施します")
            return True  # ノイズキャンセルを行う
        elif choice == '2':
            print("ノイズキャンセルは実施しません")
            return False  # ノイズキャンセルを行わない
        else:
            print("無効な選択です。もう一度入力してください。")

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

# 1ピクセルあたりの長さ（cm）
cm_per_pixel = 1.0 / pixel_per_cm

# 補正用の四角形の面積（2cm × 2cm）
square_area_cm2 = 2 * 2  # 2cm × 2cm = 4 cm²
square_area_pixels = (pixel_per_cm * 2) ** 2  # ピクセル数での四角形の面積

# ノイズキャンセルモードの選択
noise_cancellation = select_mode()
if noise_cancellation:
    min_area = 100
else:
    min_area = 0

# 入力画像フォルダと出力結果フォルダのパス
input_folder = 'images'
output_folder = 'results'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 実行時刻を取得して、フォルダ名に使用
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# csv書き込みようにも実行日を取得
execution_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

# CSVファイルを作成し、実行日と閾値を書き込む
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # 実行日を書き込む
    csv_writer.writerows([
        ['Execution Date', execution_date],
        ['Program Type', 'nomal'],
        ])
    csv_writer.writerow([]) #空行
    
    # 複数の閾値を書き込む
    csv_writer.writerows([
        ['SETTING'],
        ['Grayscale Threshold', thresholds['grayscale_threshold']],
        ['Lower Hue', thresholds['lower_hue']],
        ['Lower Saturation', thresholds['lower_saturation']],
        ['Lower Value', thresholds['lower_value']],
        ['Upper Hue', thresholds['upper_hue']],
        ['Upper Saturation', thresholds['upper_saturation']],
        ['Upper Value', thresholds['upper_value']],
        ['min_area',min_area]
    ])
    csv_writer.writerow([]) #空行

    #校正情報を書き込む
    csv_writer.writerows([
        ['CORRECTION'],
        ['Reference Square Area (pixels)', square_area_pixels],
        ['Reference Square Area (cm^2)', square_area_cm2],
        ['Pixel per cm', cm_per_pixel]
    ])
    csv_writer.writerow([]) #空行

    csv_writer.writerow(['RESULTS'])
    # ヘッダーを書き込む
    csv_writer.writerows([
        ['RESULTS'],
        ['File', 'Grayscale','','HSV','','Total Image', '', ],
        [
        'Filename', 
        'Grayscale Plant Area (pixels)', 
        'Grayscale Plant Area (cm^2)', 
        'HSV Plant Area (pixels)',
        'HSV Plant Area (cm^2)',
        'Total Image Area (pixels)', 
        'Total Image Area (cm^2)', 
        ]
    ])

##################################
# ここから各画像に対して実施する計算
##################################

# 画像フォルダからすべての画像ファイルを取得
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # 画像を読み込む
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:   # 画像が読み込まれていない場合、スキップ
        print(f"Error: Image '{image_file}' not found or cannot be opened.")
        continue

    file_name_without_ext = os.path.splitext(image_file)[0] # 拡張子を除いたファイル名部分だけを取得

    # 画像全体のピクセル数と面積を計算
    total_pixels = image.shape[0] * image.shape[1]
    total_area_cm2 = total_pixels * (cm_per_pixel ** 2)  # ピクセル数から面積に変換

    ####
    # ここから画像処理
    # ####

    # グレースケール処理
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #グレースケール変換
    _, threshold_image = cv2.threshold(gray_image, thresholds['grayscale_threshold'], 255, cv2.THRESH_BINARY)   # グレースケールの閾値処理

    # HSV画像に変換して緑色抽出
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([thresholds['lower_hue'], thresholds['lower_saturation'], thresholds['lower_value']])
    upper_green = np.array([thresholds['upper_hue'], thresholds['upper_saturation'], thresholds['upper_value']])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    if noise_cancellation:
        # グレースケール画像に対してのノイズ除去
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # グレースケール画像の小さい領域を除去するための輪郭抽出
        min_area_threshold = min_area  # 小さい領域を削除するための閾値 (例: 面積が100ピクセル以下の領域を削除)
        # 小さい領域を削除した新しいグレースケールマスクを作成
        cleaned_threshold_image = np.zeros_like(threshold_image)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area_threshold:
                # 小さい領域でなければ新しいグレースケールマスクに描画
                cv2.drawContours(threshold_image, [contour], -1, 255, thickness=cv2.FILLED)

        # HSVのノイズ除去処理
        # モルフォロジー ##モルフォジー処理についてはネット参照のこと
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #クローズ処理（膨張→収縮）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) #オープン処理（収縮→膨張）
        # 小さい領域を除去するための輪郭抽出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #  小さい領域の削除
        min_area_threshold = min_area  # 小さい領域を削除するための閾値 (例: 面積が100ピクセル以下の領域を削除)
        mask = np.zeros_like(mask)  # 小さい領域を削除した新しいマスクを作成
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area_threshold:
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)    # 小さい領域でなければ新しいマスクに描画

    #####
    # ここから面積計算
    # #####

    # 面積計算（グレースケール）
    area_gray_pixels = np.sum(threshold_image == 255)
    area_gray_cm2 = area_gray_pixels / (pixel_per_cm ** 2)  # ピクセル数から平方センチメートルに変換

    # 面積計算（緑抽出）
    area_green_pixels = np.sum(mask == 255)
    area_green_cm2 = area_green_pixels / (pixel_per_cm ** 2)  # ピクセル数から平方センチメートルに変換

    # グレースケール境界画像に面積を表示（グレースケール画像上）
    gray_image_with_text = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)  # グレースケールを3チャンネルに変換
    cv2.putText(gray_image_with_text, f'Grayscale Area: {area_gray_cm2:.2f} cm^2', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    # 緑のマスク境界画像に面積を表示（HSVマスク画像上）
    mask_image_with_text = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # マスクを3チャンネルに変換
    cv2.putText(mask_image_with_text, f'Green Area: {area_green_cm2:.2f} cm^2', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    # 元画像にもテキストを表示（元画像上）
    cv2.putText(image, f'Original Image', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    # グレースケール、HSVマスク、元画像を横に結合して表示する
    combined_image = np.hstack((cv2.cvtColor(image, cv2.COLOR_BGR2RGB), gray_image_with_text, mask_image_with_text))

    # 結果の画像を保存（横並びの画像）
    result_image_path = os.path.join(output_folder_with_time_image, f'result_{file_name_without_ext}.jpg')
    plt.figure(figsize=(20, 10))
    plt.imshow(combined_image)
    plt.axis('off')  # 軸を非表示にする
    plt.tight_layout()
    plt.savefig(result_image_path)
    plt.close()

    # 画像ごとの結果をCSVに追記
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            image_file, 
            area_gray_pixels,
            area_gray_cm2,
            area_green_pixels, 
            area_green_cm2,
            total_pixels, 
            total_area_cm2, 
        ])

    print(f"Processed and saved: {result_image_path}")