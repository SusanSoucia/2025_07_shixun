from engine import FaceRecognizer
from flask import Blueprint,current_app,render_template,request,redirect,url_for,jsonify,flash,Response
from werkzeug.utils import secure_filename
import os,cv2,numpy as np
import base64,time
from .utils.camera import Camera

# --------------------初始化-----------------------#
engine = FaceRecognizer()
main_bp = Blueprint("main",__name__)
_camera = None

def warm_up():
    # app 启动时执行预热
    dummy = np.zeros((112,112,3), dtype=np.float32)
    engine.recognize(dummy, aligned=True)

warm_up()


# --------------------工具-------------------------#
ALLOWED_EXT = {"jpg","jpeg","png","webp"}

def allowed_file(filename: str)->bool:
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

def file_to_img_bytes(file_storage):
    bytes_data = file_storage.read()
    img = cv2.imdecode(np.frombuffer(bytes_data,np.uint8),cv2.IMREAD_COLOR)
    return img,bytes_data

def get_camera():
    global _camera
    if _camera is None:
        _camera = Camera(engine,skip=20)

    return _camera






# --------------------路由-------------------------#
@main_bp.route("/")
def index():
    return render_template("indexNew.html")

@main_bp.route("/identify",methods = ["GET","POST"])
def identify():
    if request.method == "GET":
        return render_template("identifyNew.html")
    
    file = request.files.get("image")
    if not file or not allowed_file(file.filename):
        flash("格式错误")
        return redirect(url_for("main.identify"))
    
    img_bgr, img_bytes = file_to_img_bytes(file)
    if img_bgr is None:
        flash("图像解析失败")
        return redirect(url_for("main.identify"))
    
    name,score = engine.recognize(img_bgr)
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return render_template(
        "result.html",
        name = name if name else "unknown",
        score=score,
        img_data=img_b64,
    )
    
    
    
@main_bp.route("/register",methods =["GET","POST"])
def register():
    if request.method == "GET":
        return render_template("registerNew.html")
    
    name = request.form.get("name","").strip()
    if not name:
        flash("姓名不能为空")
        return redirect(url_for("main.register"))
    
    files = request.files.getlist("images")
    print(f"收到{len(files)}张图片")

    if not files:
        flash("请至少选择一张图片")
        return redirect(url_for("main.register"))
    
    ok,fail = 0,0

    for f in files:
        if not allowed_file(f.filename):
            fail+=1
            continue
        img_bgr,_ = file_to_img_bytes(f)
        if img_bgr is None:
            fail+=1
            continue
        try:
            engine.register(img_bgr,name=name,single=False)
            ok+=1
        except Exception as e:
            print("注册失败:",e)
            fail+=1

    if name in engine.vector_db and len(engine.vector_db[name])>0:
        engine.compute_mean_vector(name,engine.vector_db)
    else:
        flash(f"⚠️ 无有效人脸向量，无法计算平均向量")
        return redirect(url_for("main.register"))
    if ok:
        flash(f"✔ 已成功添加 {ok} 张图片到身份『{name}』")
    if fail:
        flash(f"⚠ 有 {fail} 张图片注册失败，请检查格式或人脸质量")

    return redirect(url_for("main.register"))
    

@main_bp.route("/live")
def live():
    return render_template("liveNew.html")

@main_bp.route("/video_feed")
def video_feed():
    cam = get_camera()
    def gen():
        while True:
            frame = cam.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.01)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@main_bp.route("/stop_cam", methods=["POST"])
def stop_cam():
    global _camera
    if _camera is not None:
        _camera.stop()          # cap.release()，running=False
        _camera = None
    return ("", 204)            # 空响应

