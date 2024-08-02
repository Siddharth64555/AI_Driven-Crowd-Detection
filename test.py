import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import asyncio
from telegram import Bot
from concurrent.futures import ThreadPoolExecutor
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# Replace with your actual bot token and chat ID
TBT = 'BOT-ID'
CID = 'RECIPIENT-ID'

bot = Bot(token=TBT)
executor = ThreadPoolExecutor()

model = YOLO('best.pt')
model = model.to('cuda')

alert_sent = False
alert_sent_time = 0  # Track the time when the alert was last sent
alert_delay = 60  # Delay in seconds between alerts
count = 0  # Initialize count


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('cr.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

area1 = [(11, 9), (1008, 3), (1016, 485), (12, 484)]


async def send_telegram_message(message):
    try:
        await bot.send_message(chat_id=CID, text=message)
        logging.info(f"Message sent: {message}")
    except Exception as e:
        logging.error(f"Failed to send message: {e}")


async def process_frame():
    global alert_sent, alert_sent_time, count  # Declare these as global variables
    loop = asyncio.get_event_loop()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        count += 1
        if count % 3 != 0:
            continue
        frame = cv2.resize(frame, (1020, 500))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        a = results[0].boxes.data.cpu().numpy()
        px = pd.DataFrame(a).astype("float")

        cr = 0
        list1 = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            w, h = x2 - x1, y2 - y1
            result = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)
            if result >= 0:
                cvzone.cornerRect(frame, (x1, y1, w, h), 6, 6)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cvzone.putTextRect(frame, 'person', (x1, y1), 1, 1)
                list1.append(cx)

        cr = len(list1)
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
        cvzone.putTextRect(frame, f'c1:-{cr}', (50, 60), 1, 1)

        current_time = time.time()
        if cr > 4:
            if not alert_sent or (current_time - alert_sent_time > alert_delay):
                alert_sent = True
                alert_sent_time = current_time
                # Use run_in_executor to avoid blocking the event loop
                loop.run_in_executor(executor, lambda: asyncio.run(
                    send_telegram_message(f"Alert! {cr} people detected in the video.")))
        else:
            alert_sent = False

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(process_frame())
