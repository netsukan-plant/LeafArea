import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込む
image = cv2.imread('sample.jpeg')

# 画像が読み込まれていない場合、エラーメッセージを出力
if image is None:
    print("Error: Image not found or cannot be opened.")
    exit()

# BGR画像をHSV画像に変換
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 緑色の範囲を定義 (色相: 35~85, 彩度と明度の範囲を調整)
lower_green = np.array([35, 40, 40])  # 緑色の下限
upper_green = np.array([85, 255, 255])  # 緑色の上限

# 緑色の範囲内のピクセルを抽出するマスクを作成
mask = cv2.inRange(hsv_image, lower_green, upper_green)

# 元の画像にマスクを適用して、緑色の部分だけを抽出
green_extracted = cv2.bitwise_and(image, image, mask=mask)

# 結果を表示
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(green_extracted, cv2.COLOR_BGR2RGB))
plt.title('Green Color Extracted')

plt.show()
