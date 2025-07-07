import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Input,BatchNormalization
from tensorflow.keras.applications import ResNet50
from .layers import ArcMarginPenaltyLogists
from tensorflow.keras.callbacks import TensorBoard


# %%
tf.config.list_physical_devices('GPU')

# %% [markdown]
# # ArcFace tensorflow风格实现

# %%
# ------------------------------------------------------------------
# 1️⃣   Backbone
# ------------------------------------------------------------------
def resnet50_backbones(input_shape = (112,112,3), use_pretrain=True,name = 'resnet50'):
    """返回 ResNet50 backbone 的输出特征图"""
    weights = "imagenet" if use_pretrain else None
    inputs = Input(shape = input_shape,name = 'backbone_input')
    base = ResNet50(include_top=False,
                    weights=weights,
                    input_tensor=inputs)
    outputs = base.output
    return Model(inputs, outputs, name=name)

# ------------------------------------------------------------------
# 2️⃣   Embedding Head
# ------------------------------------------------------------------
def embeddingLayer(embd_shape=512, w_decay=5e-4, name="OutputLayer"):
    x_in = Input(shape=(4, 4, 2048), name="output_input")  # 明确指定形状
    x = BatchNormalization(name="batch_normalization")(x_in)
    x = Dropout(0.5, name="dropout")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(
        embd_shape,
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(w_decay),
        name="dense",
    )(x)
    x = BatchNormalization(name="batch_normalization_1")(x)
    return Model(x_in, x, name=name)


# ------------------------------------------------------------------
# 3️⃣   ArcFace Head
# ------------------------------------------------------------------
def ArcHead(num_classes, margin=0.5, logist_scale=64, name="ArcHead"):
    """Embedding + labels -> ArcFace logits"""
    def _head(embd, labels):
        logits = ArcMarginPenaltyLogists(
            num_classes=num_classes,
            margin=margin,
            logist_scale=logist_scale,
            name=name,
        )(embd, labels)
        return logits
    return _head

# ------------------------------------------------------------------
# 4️⃣   ArcFace Model Builder
# ------------------------------------------------------------------
def ArcFaceModel(
    size=112,
    channels=3,
    num_classes=None,
    margin=0.5,
    logist_scale=64,
    embd_shape=512,
    w_decay=5e-4,
    use_pretrain=True,
    training=False,
    name="arcface_model",
):
    """
    training=False: image -> embedding
    training=True : [image, label] -> logits
    """
    images = Input([size, size, channels], name="image")

    # Backbone
    backbone_model = resnet50_backbones((size,size,channels), use_pretrain=use_pretrain)
    embedding_model = embeddingLayer(embd_shape, w_decay)

    feat = backbone_model(images)
    # Embedding
    embd = embedding_model(feat)

    if training:
        if num_classes is None:
            raise ValueError("`num_classes` must be set when `training=True`")
        labels = Input(shape=(), dtype="int32", name="label")
        logits = ArcHead(num_classes, margin, logist_scale)(embd, labels)
        return Model([images, labels], logits, name=name)
    else:
        return Model(images, embd, name=name)


# %%
# ------------------------------------------------------------------
# 5️⃣   使用示例：加载权重
# ------------------------------------------------------------------
"""
if __name__ == "__main__":
    WEIGHTS = "/home/wucwz/Learning/faceai/ArcFace-Res50.h5"

    # ▸ ① 推理模型：只输出 512‑D 向量
    infer_model = ArcFaceModel(size=112, training=False)
    infer_model.load_weights(WEIGHTS, by_name=True, skip_mismatch=False)
    print("Inference model loaded ✓")

    # ▸ ② 训练模型：需要 num_classes & labels 输入

    dataset = datasetAlt.Dataset('/home/wucwz/Learning/faceai/dataset/facescrub',img_size=112,batch_size=32)
    train_ds, val_ds = dataset.for_arcface()

    train_model = ArcFaceModel(size=112,
                               num_classes=dataset.num_classes,
                               training=True)
    train_model.load_weights(WEIGHTS, by_name=True, skip_mismatch=False)
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # train_model.load_weights(WEIGHTS, by_name=True, skip_mismatch=True)

    EPOCHS = 20
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('arcface_best.h5', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.1),
        tensorboard_callback,
    ]

    train_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )


    print("Train model ✓")
"""