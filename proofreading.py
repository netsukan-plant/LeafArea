import cv2

# 2cm×2cmの黒い四角形を認識し、1cmあたりのピクセル数を計算する関数
def calculate_pixel_per_cm(reference_image_path):
    reference_image = cv2.imread(reference_image_path)

    if reference_image is None:
        print(f"Error: Reference image '{reference_image_path}' not found.")
        exit()

    # グレースケール画像に変換して閾値処理を行い、黒い四角形を検出
    gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    _, threshold_reference = cv2.threshold(gray_reference, 127, 255, cv2.THRESH_BINARY_INV)

    # 輪郭を検出
    contours, _ = cv2.findContours(threshold_reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最大の輪郭を取得
    largest_contour = max(contours, key=cv2.contourArea)

    # 輪郭の外接矩形を取得
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 2cmの幅を持つ黒い四角形の1cmあたりのピクセル数を計算
    pixel_per_cm = w / 2.0  # 2cmの幅なので1cmはその半分
    print(f"Calculated pixel per cm: {pixel_per_cm} pixels/cm")
    return pixel_per_cm
