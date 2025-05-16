from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from train import create_dataset  # ✅ เปลี่ยนตรงนี้

# ✅ ใช้ฟังก์ชันจาก train.py เพื่อสร้าง train/val generators
train_generator, val_generator = create_dataset('dataset')  # หรือใส่ path ที่ถูกต้อง

# สร้างโมเดล MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Compile & train
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save model
model.save("solution_model.h5")
