import cv2
import numpy as np
import csv
import os

# 保存するCSVファイルのパス
thresholds_file = 'thresholds.csv'

# サンプル画像を読み込む（ここでは sample.jpg と仮定）
sample_image_path = 'sample.jpg'
image = cv2.imread(sample_image_path)

# 画像が存在しない場合はエラーメッセージ
if image is None:
    print(f"Error: Sample image '{sample_image_path}' not found.")
    exit()

# 閾値をCSVから読み込む関数
def load_thresholds(file_path):
    # デフォルトの初期値
    thresholds = {
        'grayscale_threshold': 127,
        'lower_hue': 0,
        'lower_saturation': 50,
        'lower_value': 50,
        'upper_hue': 180,
        'upper_saturation': 255,
        'upper_value': 255
    }
    
    # CSVファイルが存在するか確認
    if os.path.exists(file_path):
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            # 最初の行を読み込む
            for row in csv_reader:
                thresholds['grayscale_threshold'] = int(row['grayscale_threshold'])
                thresholds['lower_hue'] = int(row['lower_hue'])
                thresholds['lower_saturation'] = int(row['lower_saturation'])
                thresholds['lower_value'] = int(row['lower_value'])
                thresholds['upper_hue'] = int(row['upper_hue'])
                thresholds['upper_saturation'] = int(row['upper_saturation'])
                thresholds['upper_value'] = int(row['upper_value'])
                break  # 1行目だけ読み込む
    return thresholds

# 閾値をCSVに保存する関数
def save_thresholds(file_path, grayscale_threshold, lower_hue, lower_saturation, lower_value, upper_hue, upper_saturation, upper_value):
    # CSVファイルに現在のトラックバーの値を保存
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['grayscale_threshold', 'lower_hue', 'lower_saturation', 'lower_value', 'upper_hue', 'upper_saturation', 'upper_value'])
        writer.writerow([grayscale_threshold, lower_hue, lower_saturation, lower_value, upper_hue, upper_saturation, upper_value])
    print(f"Thresholds saved to {file_path}")

# 初期値をCSVから読み込む
initial_values = load_thresholds(thresholds_file)

# 閾値を設定するトラックバーのコールバック関数
def nothing(x):
    pass

# OpenCVのウィンドウを作成
cv2.namedWindow('Threshold Adjustments')

# トラックバーを作成し、CSVから読み込んだ初期値をセット
cv2.createTrackbar('Grayscale Threshold', 'Threshold Adjustments', initial_values['grayscale_threshold'], 255, nothing)
cv2.createTrackbar('Lower Hue', 'Threshold Adjustments', initial_values['lower_hue'], 180, nothing)
cv2.createTrackbar('Lower Saturation', 'Threshold Adjustments', initial_values['lower_saturation'], 255, nothing)
cv2.createTrackbar('Lower Value', 'Threshold Adjustments', initial_values['lower_value'], 255, nothing)
cv2.createTrackbar('Upper Hue', 'Threshold Adjustments', initial_values['upper_hue'], 180, nothing)
cv2.createTrackbar('Upper Saturation', 'Threshold Adjustments', initial_values['upper_saturation'], 255, nothing)
cv2.createTrackbar('Upper Value', 'Threshold Adjustments', initial_values['upper_value'], 255, nothing)

# 初期状態の表示用の変数
prev_threshold = -1
prev_lower_hue = -1
prev_lower_saturation = -1
prev_lower_value = -1
prev_upper_hue = -1
prev_upper_saturation = -1
prev_upper_value = -1

# 毎フレーム更新する処理
while True:
    # 現在のトラックバーの値を取得
    grayscale_threshold = cv2.getTrackbarPos('Grayscale Threshold', 'Threshold Adjustments')
    lower_hue = cv2.getTrackbarPos('Lower Hue', 'Threshold Adjustments')
    lower_saturation = cv2.getTrackbarPos('Lower Saturation', 'Threshold Adjustments')
    lower_value = cv2.getTrackbarPos('Lower Value', 'Threshold Adjustments')
    upper_hue = cv2.getTrackbarPos('Upper Hue', 'Threshold Adjustments')
    upper_saturation = cv2.getTrackbarPos('Upper Saturation', 'Threshold Adjustments')
    upper_value = cv2.getTrackbarPos('Upper Value', 'Threshold Adjustments')

    # もしトラックバーの値が変わった場合のみ再計算
    if (grayscale_threshold != prev_threshold or lower_hue != prev_lower_hue or
        lower_saturation != prev_lower_saturation or lower_value != prev_lower_value or
        upper_hue != prev_upper_hue or upper_saturation != prev_upper_saturation or
        upper_value != prev_upper_value):
        
        # グレースケールの閾値処理
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(gray_image, grayscale_threshold, 255, cv2.THRESH_BINARY)

        # HSVの下限と上限でマスクを作成
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([lower_hue, lower_saturation, lower_value])
        upper_green = np.array([upper_hue, upper_saturation, upper_value])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # 動的に更新する画像（グレースケールとマスク）
        combined_image = np.hstack((cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('Threshold Adjustments', combined_image)

        # 現在のトラックバーの値を保持
        prev_threshold = grayscale_threshold
        prev_lower_hue = lower_hue
        prev_lower_saturation = lower_saturation
        prev_lower_value = lower_value
        prev_upper_hue = upper_hue
        prev_upper_saturation = upper_saturation
        prev_upper_value = upper_value

    # キー入力を処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 現在のトラックバーの値をCSVに保存
        save_thresholds(thresholds_file, grayscale_threshold, lower_hue, lower_saturation, lower_value, upper_hue, upper_saturation, upper_value)
        print('変更を保存しました')

# ウィンドウを閉じる
cv2.destroyAllWindows()