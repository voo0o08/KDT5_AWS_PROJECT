import pyautogui
import time

# 현재 사용하는 모니터의 해상도 출력
print(pyautogui.size())

while True:
    # 현재 마우스 커서의 위치 출력
    print(pyautogui.position())
    time.sleep(3)
