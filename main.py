import torch
import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import math
import cv2
from aiohttp import web
from av import VideoFrame
import aiohttp_cors
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from ultralytics import YOLO
import pyrebase
import threading
from dotenv import load_dotenv

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

# firebaseConfig = {
#   "apiKey": "AIzaSyAgo9a0sshqOGo9qZM7tfSBNMN3qiEyUK0",
#   "authDomain": "bemyeyes-3c6ef.firebaseapp.com",
#   "databaseURL": "https://bemyeyes-3c6ef-default-rtdb.firebaseio.com",
#   "projectId": "bemyeyes-3c6ef",
#   "storageBucket": "bemyeyes-3c6ef.appspot.com",
#   "messagingSenderId": "557432449633",
#   "appId": "1:557432449633:web:b1ab371c0a17270dd91820",
#   "measurementId": "G-L2JTBP93Z0",
#   "serviceAccount":"Firebase_Service_Account_Keys.json"
# }

firebaseConfig = {
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
    "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN")
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()
storage = firebase.storage()
auth = firebase.auth()

# model = YOLO('model_f.pt')
model2 = YOLO('veg.pt')
model =YOLO('model_veg_toy.pt')
model.to('cuda')

def storage_function():
    img_file_name = 'crop_img.jpg'
    storage.child("idf").child("imgs").child(img_file_name).put(img_file_name)
    img_url = storage.child("idf").child("imgs").child(img_file_name).get_url(None)
    db.child("Images").child("childName").set(img_url) #check this

def db_functions(position,audio_played_leftup,audio_played_rightup,audio_played_leftdown,audio_played_righdown,audio_played_center,currentClass):
        if "up" in position and "left" in position and not audio_played_leftup:
            # play_audio("ul.mp3")
            db.child("Audio").child("playThis").set("ul.mp3")
            audio_played_leftup = True
            audio_played_rightup = False
            audio_played_leftdown = False
            audio_played_righdown = False
            audio_played_center = False

        if "up" in position and "right" in position and not audio_played_rightup:
            # play_audio("ur.mp3")
            db.child("Audio").child("playThis").set("ur.mp3")
            audio_played_rightup = True
            audio_played_leftup = False
            audio_played_leftdown = False
            audio_played_righdown = False
            audio_played_center = False

        if "down" in position and "left" in position and not audio_played_leftdown:
            # play_audio("dl.mp3")
            db.child("Audio").child("playThis").set("dl.mp3")
            audio_played_leftdown = True
            audio_played_leftup = False
            audio_played_rightup = False
            audio_played_righdown = False
            audio_played_center = False

        if "down" in position and "right" in position and not audio_played_righdown:
            # play_audio("dr.mp3")
            db.child("Audio").child("playThis").set("dr.mp3")
            audio_played_righdown = True
            audio_played_leftup = False
            audio_played_leftup = False
            audio_played_rightup = False
            audio_played_leftdown = False
            audio_played_center = False

        if "center" in position and not audio_played_center:
            # play_audio("center.mp3")
            db.child("Audio").child("Center").set("center")
            db.child("Audio").child("playThisObject").set(currentClass)
            audio_played_center = True
            audio_played_righdown = False
            audio_played_leftup = False
            audio_played_leftup = False
            audio_played_rightup = False
            audio_played_leftdown = False


class VideoTransformTrack(MediaStreamTrack):

    kind = "video"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track

    async def recv(self):
        audio_played = False
        audio_played_leftup = False
        audio_played_rightup = False
        audio_played_leftdown = False
        audio_played_righdown = False
        audio_played_center = False
        position = ""
        currentClass = ""
        distance = 0.00
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        # Get the frame width and height
        frame_width, frame_height = img.shape[:2]
        # width = 1200
        # height = 1000 
        width = frame_width
        height = frame_height 
        dim = (width, height)
        
        # resize image
        # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # frame_height, frame_width = img.shape[:2]

        # Format the size text
        size_text = f"Width: {frame_width}, Height: {frame_height}"
        # frame_width1, frame_height1 = img.shape[:2]
        # size_text1 = f"Width: {frame_width1}, Height: {frame_height1}"

        # new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        # new_frame.pts = frame.pts
        # new_frame.time_base = frame.time_base
        # return new_frame
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 0, 0)

        # classNames = ["apple", "banana", "dragon_fruit", "guava", "oren", "pear", "pineapple", "sugar_apple"]
        classNamesVeg = ['beet', 'bell_pepper', 'cabbage', 'carrot', 'cucumber', 'egg', 'eggplant', 'garlic', 'onion', 'potato', 'tomato', 'zucchini']
        classNames = ['apple', 'bell-pepper', 'carrot', 'pineapple']
        
        results = model.track(img, persist=True,conf=0.7)
        # annotated_frame = results[0].plot()
        
        for r in results:
            boxes = r.boxes
            if boxes.is_track:
                x1, y1, x2, y2 = boxes[0].xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                classNumber = int(boxes[0].cls[0])
                currentClass = classNames[classNumber]

                ball_center_x = x1 + w // 2
                ball_center_y = y1 + h // 2

                horizontal_distance = ball_center_x - height // 2
                vertical_distance = ball_center_y - width // 2

                if horizontal_distance > 0:
                    position += "right "
                elif horizontal_distance < 0:
                    position += "left "

                if vertical_distance > 0:
                    position += "down"
                elif vertical_distance < 0:
                    position += "up"

                if position == "":
                    position = "center"

                distance = math.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)

                # cv2.line(img, (frame_height // 2, frame_width // 2), (int(ball_center_x), int(ball_center_y)), (0, 0, 255), 2)
                try:
                    cv2.circle(img, (ball_center_x, ball_center_y), 5, (255, 0, 255), cv2.FILLED)
                except:
                    pass

                if distance < 35:
                    # cv2.putText(img, "center", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    position = "center"
                    crop_img = img[y1:y2, x1:x2]
                    img_file_name = 'crop_img.jpg'
                    cv2.imwrite(img_file_name, crop_img)
                    # storage_function(crop_img)
                    y = threading.Thread(target=storage_function)
                    y.start()
                
                annotated_frame = results[0][0].plot(conf=False,labels=False,probs=False)
                # print(f"results 0 len: {len(results[0])}")
                # print(f"results 0 : {results[0]}")
                # print(f"box 0 : {boxes[0]}")
            else:
                # annotated_frame = img
                results2 = model2.track(img, persist=True,conf=0.8)
                for r in results2:
                    boxes = r.boxes
                    if boxes.is_track:
                        x1, y1, x2, y2 = boxes[0].xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1

                        classNumber = int(boxes[0].cls[0])
                        currentClass = classNamesVeg[classNumber]

                        ball_center_x = x1 + w // 2
                        ball_center_y = y1 + h // 2

                        horizontal_distance = ball_center_x - height // 2
                        vertical_distance = ball_center_y - width // 2

                        if horizontal_distance > 0:
                            position += "right "
                        elif horizontal_distance < 0:
                            position += "left "

                        if vertical_distance > 0:
                            position += "down"
                        elif vertical_distance < 0:
                            position += "up"

                        if position == "":
                            position = "center"

                        distance = math.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)

                        # cv2.line(img, (frame_height // 2, frame_width // 2), (int(ball_center_x), int(ball_center_y)), (0, 0, 255), 2)
                        try:
                            cv2.circle(img, (ball_center_x, ball_center_y), 5, (255, 0, 255), cv2.FILLED)
                        except:
                            pass

                        if distance < 35:
                            # cv2.putText(img, "center", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            position = "center"
                            crop_img = img[y1:y2, x1:x2]
                            img_file_name = 'crop_img.jpg'
                            cv2.imwrite(img_file_name, crop_img)
                            # storage_function(crop_img)
                            y = threading.Thread(target=storage_function)
                            y.start()
                        
                        annotated_frame = results2[0][0].plot(conf=False,labels=False,probs=False)
                    else:
                        annotated_frame = img

        # annotated_frame = img

        cv2.circle(annotated_frame, (height // 2, width // 2), 5, (0, 0, 255), -1)
        # Center horizontal line
        cv2.line(annotated_frame, (0, width // 2), (height, width // 2), color, 2)
        # Center vertical line
        cv2.line(annotated_frame, (height // 2, 0), (height // 2, width), color, 2)

        # Put text on the image
        cv2.putText(annotated_frame, size_text, (10, 30), font, font_scale, color, 1)
        # cv2.putText(annotated_frame, size_text1, (20, 60), font, font_scale, color, 1)

        cv2.imshow("Transformed Image", annotated_frame)
        cv2.waitKey(1)

        # db_functions(position,audio_played_leftup,audio_played_rightup,audio_played_leftdown,audio_played_righdown,audio_played_center,currentClass)
        x = threading.Thread(target=db_functions, args=(position,audio_played_leftup,audio_played_rightup,audio_played_leftdown,audio_played_righdown,audio_played_center,currentClass))
        x.start()

        new_frame = VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
        # return frame

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track)
                )
            )

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


app = web.Application()
cors = aiohttp_cors.setup(app)
app.on_shutdown.append(on_shutdown)
app.router.add_post("/offer", offer)

for route in list(app.router.routes()):
    cors.add(route, {
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument(
        "--host", default="192.168.1.4", help="Host for HTTP server (default: 0.0.0.0) my ip: 192.168.1.3"
    )
    parser.add_argument(
        "--port", type=int, default=8002, help="Port for HTTP server (default: 8080)"
    )

    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    web.run_app(
        app, access_log=None, host=args.host, port=args.port
    )
