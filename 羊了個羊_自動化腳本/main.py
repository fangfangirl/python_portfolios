import pyautogui
picture = ['1.png','2.png','3.png','4.png','5.png','6.png','7.png','8.png','9.png','10.png']
a = [[0]*9 for i in range(30)]
count = [0,0,0,0,0,0,0,0,0,0]
def openpage():
    pyautogui.PAUSE = 1
    pyautogui.hotkey('win','r')
    pyautogui.typewrite('https://solvable-sheep-game.streakingman.com/')
    pyautogui.press('enter')
    pyautogui.hotkey('ctrl', '-')

def find():
    for i in range(0,10):
        count[i] = 0
    for i in range(0,10):
        num=0
        for j in pyautogui.locateAllOnScreen(picture[i], confidence=0.98):
            a[i][num]=j
            count[i]=count[i]+1;
            num = num+1

def click(count,a,i):

    if count[i] >= 3 :
        c = count[i]//3;
        for j in range(3*c):
            loc = a[i][j]
            center = pyautogui.center(loc)
            pyautogui.click(center)
            count[i]=count[i]-1;

    if count[i]%3 == 2:
        count[i]=0
        num = 0
        for j in pyautogui.locateAllOnScreen(picture[i], confidence=0.98):
            a[i][num] = j
            count[i] = count[i] + 1;
            num = num + 1
        if count[i]%3==0 and count[i]>0:
            for j in range(3):
                loc = a[i][j]
                center = pyautogui.center(loc)
                pyautogui.click(center)
                count[i] = count[i] - 1;

def click2(count,a,i):
    for j in range(2):
        loc = a[i][j]
        center = pyautogui.center(loc)
        pyautogui.click(center)
        count[i] = count[i] - 1;
    count[i] =0

    num = 0
    for j in pyautogui.locateAllOnScreen(picture[i], confidence=0.98):
        a[i][num] = j
        count[i] = count[i] + 1;
        num = num + 1

    if count[i] >=1:
        for j in range(1):
            loc = a[i][j]
            center = pyautogui.center(loc)
            pyautogui.click(center)
            count[i] = count[i] - 1;

def click3(count,a,i):
    for j in range(1):
        loc = a[i][j]
        center = pyautogui.center(loc)
        pyautogui.click(center)
        count[i] = count[i] - 1;
    count[i] =0
    num = 0
    for j in pyautogui.locateAllOnScreen(picture[i], confidence=0.98):
        a[i][num] = j
        count[i] = count[i] + 1;
        num = num + 1
    three = 1
    while count[i] >0:
        for j in range(1):
            loc = a[i][j]
            center = pyautogui.center(loc)
            pyautogui.click(center)
            count[i] = count[i] - 1;
        three += 1
        num = 0
        for j in pyautogui.locateAllOnScreen(picture[i], confidence=0.98):
            a[i][num] = j
            count[i] = count[i] + 1;
            num = num + 1
        if three >= 3:
            break

#以下為主程式

openpage() #打開網頁
find() #搜尋頁面中的圖片並分類

flag= 0 #判斷是否停止
for i in range(0, 10):
    if count[i]!=0:
        flag=1



while (flag == 1):  # 非停止
    index = 0
    for i in range(0, 10):  # 檢測是否整個場面上每種圖片只有一個
        if count[i] == 1:
            index = 1
    for i in range(0, 10):  # 檢測是否整個場面上每種圖片只有2個
        if count[i] == 2:
            index = 2
    for i in range(0, 10):  # 檢測是否整個場面上圖片有超過3個的
        if count[i] >= 3:
            index = 0

    if index == 0:
        for i in range(0,10):
            if count[i]>=3:
                click(count,a,i)
    if index == 2:
        for i in range(0, 10):
            if count[i] == 2:
                click2(count, a, i)
                break
    if index == 1:
        for i in range(0, 10):
            if count[i] == 1:
                click3(count, a, i)
                break
    find()
    flag=0
    for i in range(0, 10):
        if count[i] != 0:
            flag =1

print("Game over")
