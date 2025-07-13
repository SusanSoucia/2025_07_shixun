import cv2, threading, time
import numpy as np
import os
from PIL import Image,ImageDraw,ImageFont

class Camera:
    def __init__(self, engine, skip=5,font_path='C:/Windows/Fonts/simhei.ttf',font_size = 20):
        """
        engine : 你的 FaceRecognizer 实例
        skip   : 每 N 帧做一次识别，降低耗时
        """
        self.engine = engine
        self.skip   = max(1,skip)
        self.counter = 0
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("❌ 无法打开摄像头")
        if not os.path.isfile(font_path):
            raise FileNotFoundError(f"字体文件未找到:{font_path}")
        self.font = ImageFont.truetype(font_path,font_size)
        self.raw_frame = None
        self.frame   = None
        self.running = True
        self._last_results = None
        self.lock = threading.Lock()

        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._inference,daemon=True).start()

 # ---------- ① 采帧线程 ----------
    def _reader(self):
        while self.running:
            ok,frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.raw_frame = frame.copy()
            time.sleep(0.01)

    def _draw_text_cn(self,img_bgr,text,pos,color=(0,255,0)):
        """
        用PIL库实现中文显示
        """
        img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text(pos,text,font=self.font,fill=color[::1])
        return cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR)

    def _inference(self):
        while self.running:
            with self.lock:
                frame = None if self.raw_frame is None else self.raw_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue

            self.counter+=1
            draw = frame.copy()

            if self.counter%self.skip == 0:
                results = self.engine.recognize_multi(frame) or []
                self._last_results = results

            else:
                results = getattr(self,"_last_results",[])
            
            if results is not None:
                for (x1,y1,x2,y2),name,score in results:
                    cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,0),2)
                    label = f"{name}:{score:.2f}"
                    draw = self._draw_text_cn(draw,label,(x1,max(0,y1-25)))
                with self.lock:
                    self.frame = draw
                
            time.sleep(0.01)

            """

            if results is not None:
                for (x1, y1, x2, y2), name, score in results:
                    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(draw, f"{name}:{score:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    


                with self.lock:
                    self.frame = draw
            time.sleep(0.01)
            """

    # 原视频函数
    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01); continue

            # 每 skip 帧做一次识别
            self.counter += 1
            if self.counter % self.skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.engine.recognize(rgb)
                if result:
                    self.last_result = result    # (name, score)

            # 叠加文字
            if self.last_result:
                name, score = self.last_result
                cv2.putText(frame, f"{name} {score:.2f}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)

            self.frame = frame
            time.sleep(0.01)   # 稍微让出 CPU

    # 供路由取帧
    def get_frame(self):
        if self.frame is None:
            return None
        ret, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes() if ret else None

    def stop(self):
        self.running = False
        self.cap.release()

