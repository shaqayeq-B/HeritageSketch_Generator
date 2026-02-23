import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def generate_perfect_sketch(image_path):
    # ۱. خواندن تصویر
    img = cv2.imread(image_path)
    if img is None:
        print("تصویر پیدا نشد! لطفا مسیر فایل را چک کنید.")
        return None
    
    # ۲. کاهش نویز هوشمند (حذف بافت دیوار)
    # استفاده از دو مرحله فیلتر برای تخت کردن رنگ دیوارها
    inter = cv2.bilateralFilter(img, 15, 85, 85)
    gray = cv2.cvtColor(inter, cv2.COLOR_BGR2GRAY)
    
    # ۳. تقویت تضاد نوری برای جدا شدن پنجره‌ها از دیوار
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    
    # ۴. استفاده از Adaptive Thresholding به جای Canny
    # این متد لبه‌های ساختاری را بسیار بهتر از Canny حفظ می‌کند
    line_art = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 21, 10)
    
    # ۵. تمیزکاری مورفولوژیک (حذف نقاط سیاه تک افتاده روی دیوار)
    kernel = np.ones((2,2), np.uint8)
    # حذف نویزهای ریز (سنگ‌ریزه) و پیوسته کردن خطوط پنجره
    line_art = cv2.morphologyEx(line_art, cv2.MORPH_OPEN, kernel)
    line_art = cv2.medianBlur(line_art, 3) 

    return line_art 

# --- اجرا در VS Code ---
# نام فایل عکس خود را اینجا بنویسید
image_path = "your_image.jpg" 

result = generate_perfect_sketch(image_path) 

if result is not None:
    # نمایش خروجی نهایی
    plt.figure(figsize=(15, 15))
    plt.imshow(result, cmap='gray')
    plt.axis("off")
    plt.show() 

    # ذخیره در سیستم
    cv2.imwrite("architectural_sketch.jpg", result)
    print("فایل با موفقیت به نام architectural_sketch.jpg ذخیره شد.")
