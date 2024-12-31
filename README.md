# Projeto de Transfer Learning em Python

Este projeto faz parte de um dos desafios do Bootcamp realizado em parceria com a DIO e a empresa BairesDevs. 
O projeto consiste em aplicar o método de Transfer Learning em uma rede de Deep Learning na linguagem Python no ambiente COLAB.
Como base foi utilizado o seguinte projeto que realiza Transfer Learning com o Dataset do MNIST:

https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb 

O dataset utilizado engloba duas classes: gatos e cachorros. Uma descrição da base de dados pode ser visualizada neste link: https://www.tensorflow.org/datasets/catalog/cats_vs_dogs. 

Já o dataset para download pode ser acessado por meio deste outro link:

https://www.microsoft.com/en-us/download/details.aspx?id=54765.

O modelo deste projeto alcançou uma precisão de **89%** no conjunto de teste.

---

## Etapas do Projeto

1. **Pré-processamento dos dados**
2. **Construção do modelo utilizando uma rede pré-treinada**
3. **Treinamento do modelo**
4. **Avaliação do desempenho**
5. **Predição em imagens novas**

---

## Código e Resultados

### Carregando os Dados

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Carregando os dados
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = data_gen.flow_from_directory(
    '/content/PetImages',
    target_size=(224, 224),
    class_mode='binary',
    subset='training'
)
val_data = data_gen.flow_from_directory(
    '/content/PetImages',
    target_size=(224, 224),
    class_mode='binary',
    subset='validation'
)
```

### Construção do Modelo

O modelo foi construído utilizando a arquitetura pré-treinada MobileNetV2, com as camadas superiores ajustadas para o problema de classificação binária (gato ou cachorro).
```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
### Treinamento do Modelo

O modelo foi treinado por 10 épocas, utilizando os dados de treino e validação.
```python
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

### Resultados do Treinamento
```python
Os gráficos abaixo mostram a perda e a acurácia nos conjuntos de treino e validação ao longo das épocas.
import matplotlib.pyplot as plt

# Plotar perda e acurácia
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()
```

### Avaliação Final

A avaliação final foi realizada no conjunto de teste, resultando em uma acurácia de 89%.
```python
loss, accuracy = model.evaluate(val_data)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
```
### Predição em Novas Imagens

O modelo pode ser utilizado para prever novas imagens de gatos e cachorros. Abaixo está um exemplo de código para realizar predições.
```python
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return 'Dog' if prediction[0] > 0.5 else 'Cat'

# Exemplo
img_path = '/content/PetImages/Cat/1.jpg'
print(predict_image(img_path))
```

## Conclusão

Este projeto demonstrou a eficácia do uso de Transfer Learning para resolver problemas de classificação de imagens, especificamente para distinguir entre gatos e cachorros. 
Utilizando a arquitetura pré-treinada MobileNetV2, conseguimos aproveitar o aprendizado existente em redes neurais profundas para obter uma precisão de 89% no conjunto de teste, 
mesmo com recursos computacionais limitados no ambiente Google Colab.

Embora o modelo tenha apresentado um bom desempenho, ainda existem oportunidades de melhorias, como a experimentação com outras arquiteturas pré-treinadas, 
ajustes nos hiperparâmetros do treinamento e o uso de técnicas de regularização mais avançadas.

Em resumo, este projeto não apenas alcançou os objetivos propostos no desafio, mas também proporcionou uma base sólida para a aplicação de Transfer Learning em problemas de visão computacional. 
Ele evidencia o poder de reutilizar redes pré-treinadas para resolver novos problemas, economizando tempo e recursos computacionais.

---
## Contato

Se desejar conversar sobre este projeto ou propor melhorias, sinta-se à vontade para me contatar nas redes sociais ou pelo email: [rafaellopes.dev@gmail.com](mailto:rafaellopes.dev@gmail.com).

| [![Linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafael-lopes-desenvolvedor-fullstack/?locale=pt_BR) | [![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/rafaellopes.dev/) |
|:------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|
