# YOLO Object Detection with Voice Feedback

Este projeto utiliza o modelo YOLO para detecção de objetos em tempo real usando a webcam e fornece feedback em áudio informando as classes dos objetos detectados. O projeto é implementado com Python, usando as bibliotecas OpenCV, NumPy e pyttsx3.

---

## Funcionalidades
- Detecção de objetos em tempo real usando YOLO.
- Exibição das caixas delimitadoras e rótulos dos objetos detectados no feed da webcam.
- Feedback em áudio das classes detectadas.

---

## Pré-requisitos
- OpenCV
- NumPy
- pyttsx3
- Arquivos necessários:
  - Pesos YOLO: `yolov3-tiny.weights` ou `yolov3.weights`. você consegue através desse site https://pjreddie.com/darknet/yolo/
  - Configuração YOLO: `yolov3.cfg`.
  - Arquivo de classes COCO: `coco.names`.

---

## Explicação do Funcionamento

1. **Leitura dos Arquivos YOLO**
   - O script carrega os arquivos de configuração (`.cfg`), pesos do modelo (`.weights`) e a lista de classes (`coco.names`).
   - Estes arquivos definem o comportamento do modelo, como ele processa imagens e quais objetos ele pode detectar.

2. **Captura de Imagem da Webcam**
   - O OpenCV (`cv2`) é usado para capturar frames contínuos da webcam.
   - Cada frame é processado para detectar objetos.

3. **Pré-processamento da Imagem**
   - A imagem é convertida em um *blob*, que é um formato otimizado para entrada na rede neural.
   - O *blob* é redimensionado para 416x416 pixels e normalizado para facilitar o processamento pelo YOLO.

4. **Rede Neural YOLO**
   - A rede YOLO analisa o frame e retorna:
     - **Coordenadas das caixas delimitadoras** (bounding boxes).
     - **Classes detectadas**.
     - **Confiança** da detecção para cada objeto.

5. **Filtragem de Detecções**
   - Apenas objetos com confiança maior que 50% são considerados.
   - A técnica **Non-Maximum Suppression (NMS)** é usada para evitar caixas sobrepostas, mantendo apenas as detecções mais confiáveis.

6. **Exibição no Vídeo**
   - Para cada objeto detectado:
     - Uma caixa delimitadora é desenhada ao redor do objeto.
     - Um rótulo com o nome do objeto e a confiança é exibido acima da caixa.

7. **Feedback de Voz**
   - Usando a biblioteca `pyttsx3`, o nome da classe detectada é pronunciado.
   - Isso fornece uma experiência interativa e acessível para o usuário.

8. **Interface de Usuário**
   - O feed da webcam é exibido em uma janela chamada **"YOLO Object Detection"**.
   - O usuário pode pressionar `q` para sair do programa.

---

## Fluxo do Sistema
```mermaid
