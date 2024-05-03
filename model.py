import tensorflow as tf
import tf2onnx

tf.random.set_seed(42)

mnist = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist
x_train, x_test = x_train / 255., x_test / 255.

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters=6, kernel_size=(5, 5),
        padding="same", activation="relu", input_shape=(28, 28, 1),
    ),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation="relu"),
    tf.keras.layers.Dense(84, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

print(model.summary())

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"],
)

model.fit(
    x_train, y_train, epochs=20,
    validation_data=(x_test, y_test),
)

spec = tf.TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name="input")
model.output_names = ['output']
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=(spec,),
    output_path="model.onnx",
)

