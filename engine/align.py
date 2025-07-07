from insightface.app import FaceAnalysis
import onnxruntime as ort
import os,numpy as np
import cv2

class FaceAligner:
    def __init__(self,size = 112):
        provider = ['CUDAExecutionProvider']
        self.size = size
        self.app = FaceAnalysis(name='buffalo_1',providers=provider)
        self.app.prepare(ctx_id=0)

    def align(self,bgr,margin=0.2):
        faces = self.app.get(bgr)
        if not faces:
            return None
        faces = max(faces,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        kps = faces.kps.astype(np.float32)
        # 5-点仿射
        src  = np.array([[38.2946, 51.6963],
                         [73.5318, 51.5014],
                         [56.0252, 71.7366],
                         [41.5493, 92.3655],
                         [70.7299, 92.2041]], dtype=np.float32)
        M,_= cv2.estimateAffinePartial2D(kps,src,method=cv2.LMEDS)
        aligned = cv2.warpAffine(bgr,M,(self.size,self.size))
        return aligned.astype(np.float32)/255.0
    
    def align_multi_faces(self,bgr,det_score_threshold=0.5,margin = 0.2):
        all_faces = self.app.get(bgr)

        if not all_faces:
            return []
        valid_faces = [face for face in all_faces if face.det_score>=det_score_threshold]

        if not valid_faces: return None

        aligned_faces= []
        src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
        
        results =[]
        for face in valid_faces:
            kps = face.kps.astype(np.float32)
            M,_ = cv2.estimateAffinePartial2D(kps,src,method=cv2.LMEDS)
            aligned= cv2.warpAffine(bgr,M,(self.size,self.size))
            aligned= aligned.astype(np.float32)/255.0
            aligned_faces.append(aligned)

            x1,y1,x2,y2 = face.bbox.astype(int)
            results.append((aligned,(x1,y1,x2,y2)))
        
        return results # 返回值是一个列表

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    
    img_path = 'E:/localDL/shuangren2.png'
    bgr = cv2.imread(img_path)
    
    # 创建对齐器
    aligner = FaceAligner(size=112)
    
    # 获取所有人脸对齐结果
    aligned_faces = aligner.align_multi_faces(bgr, det_score_threshold=0.5)
    
    if not aligned_faces:
        print("未检测到任何人脸")
        exit()
    
    # 创建图像显示布局
    n_faces = len(aligned_faces)
    fig, axs = plt.subplots(1, n_faces + 1, figsize=(4*(n_faces+1), 4))
    
    # 显示原始图像（带人脸框）
    img_with_boxes = bgr.copy()
    faces = aligner.app.get(bgr)  # 获取人脸检测结果
    
    for face in faces:
        # 绘制人脸框
        bbox = face.bbox.astype(int)
        cv2.rectangle(img_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # 绘制关键点
        for kp in face.kps:
            cv2.circle(img_with_boxes, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
        
        # 添加置信度标签
        cv2.putText(img_with_boxes, f"{face.det_score:.2f}", (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 显示原始图像
    rgb_original = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    axs[0].imshow(rgb_original)
    axs[0].set_title(f"Original ({len(faces)} faces)")
    axs[0].axis("off")
    
    # 显示所有对齐后的人脸
    for i, aligned_face in enumerate(aligned_faces):
        # 转换为RGB格式用于显示
        rgb_aligned = cv2.cvtColor((aligned_face*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        
        # 显示对齐后的人脸
        axs[i+1].imshow(rgb_aligned)
        axs[i+1].set_title(f"Face {i+1}")
        axs[i+1].axis("off")
    
    plt.tight_layout()
    plt.show()