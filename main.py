# Importando bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# 1. Carregar o modelo VGG16 pré-treinado sem a camada final (fully connected)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. Congelar as camadas do modelo base para evitar que sejam treinadas
base_model.trainable = False

# 3. Criar o modelo de Transfer Learning, adicionando camadas adicionais
model = models.Sequential()

# Adicionar o modelo base (VGG16) como parte do novo modelo
model.add(base_model)

# Adicionar camadas adicionais de redes neurais para classificação
model.add(layers.Flatten())  # Flatten: transforma as saídas do modelo em um vetor
model.add(layers.Dense(128, activation='relu'))  # Camada densa com 128 neurônios
model.add(layers.Dropout(0.5))  # Dropout para evitar overfitting
model.add(layers.Dense(10, activation='softmax'))  # Camada final com 10 classes (ajuste conforme seu problema)

# 4. Compilar o modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Preparar os dados de treinamento e validação com ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza os valores dos pixels
    rotation_range=30,  # Rotaciona as imagens
    width_shift_range=0.2,  # Translação horizontal
    height_shift_range=0.2,  # Translação vertical
    shear_range=0.2,  # Aplicar cisalhamento nas imagens
    zoom_range=0.2,  # Zoom nas imagens
    horizontal_flip=True,  # Inverte horizontalmente
    fill_mode='nearest'  # Preenchimento de pixels para imagens rotacionadas ou transladadas
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Apenas normaliza os valores de teste

# 6. Carregar os dados de treino e validação
train_dir = '/path/to/train_data'  # Caminho para os dados de treino
val_dir = '/path/to/val_data'  # Caminho para os dados de validação

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Assume que temos várias classes
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 7. Treinar o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Número de épocas (ajuste conforme necessário)
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# 8. Avaliar o modelo (opcional)
# Você pode usar o modelo treinado para prever novas imagens ou avaliar seu desempenho em um conjunto de teste.

# Exemplo de como avaliar o modelo
# test_dir = '/path/to/test_data'
# test_generator = val_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32)
# test_loss, test_acc = model.evaluate(test_generator)
# print(f'Test accuracy: {test_acc}')
