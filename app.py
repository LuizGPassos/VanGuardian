import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk, ImageEnhance
import pytesseract
import pandas as pd
import numpy as np
import time

# Carregar modelo de vans
model_van = YOLO('models/vans.pt')

# Seção Prefixos
def ler_prefix(input_image):
    model_prf = YOLO('models/prefix.pt')
    results_prf = model_prf(input_image)

    #Cria as boxes e cropa o prefixo
    boxes_prf = results_prf[0].boxes.data.cpu().numpy()
    labels_prf = results_prf[0].boxes.cls.cpu().numpy()

    for i, (box, cls) in enumerate(zip(boxes_prf, labels_prf)):
        if cls == 1:
            x1, y1, x2, y2, conf, cls = box
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(input_image.width, x2 + 10)
            y2 = min(input_image.height, y2 + 10)

            prefix_crop = input_image.crop((x1, y1, x2, y2))
            text_prefix = pytesseract.image_to_string(prefix_crop, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789-')

            return text_prefix
    return ""

# Seção Placas
def ler_placa(input_image):
    #Carregar modelo de placas
    model_plc = YOLO('models/placa.pt')
    results_plc = model_plc(input_image)

    #Cria as boxes e cropa a placa
    boxes_plc = results_plc[0].boxes.data.cpu().numpy()
    labels_plc = results_plc[0].boxes.cls.cpu().numpy()

    for i, (box, cls) in enumerate(zip(boxes_plc, labels_plc)):
        x1, y1, x2, y2, conf, cls = box
        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(input_image.width, x2 + 10)
        y2 = min(input_image.height, y2 + 10)

        #Aplica Filtros na placa para melhorar a leitura
        placa_crop = input_image.crop((x1, y1, x2, y2))
        cinza = cv2.cvtColor(np.array(placa_crop), cv2.COLOR_RGB2GRAY)
        _, bnr = cv2.threshold(cinza, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dlt = cv2.dilate(bnr, None, iterations=2)
        erd = cv2.erode(dlt, None, iterations=1)

        rsz = cv2.resize(erd, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # OCR
        text_placa = pytesseract.image_to_string(
            Image.fromarray(rsz),
            lang='eng',
            config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        return text_placa
    return ""

# Verifica a lista excel
def verifica_lista(prefix, placa):
    df = pd.read_excel('lista_vans_jacarei.xlsx')
    prefixo_formatado = str(prefix).strip()
    placa_formatada = str(placa).strip()

    resultados = df.loc[(df['PREFIXO'].astype(str) == prefixo_formatado) & (df['PLACA'].astype(str) == placa_formatada)]

    if not resultados.empty:
        for index, row in resultados.iterrows():
            return f"Veiculo localizado na lista de aprovados. Proprietário: {row['NOME']}, Placa: {placa_formatada}, Prefixo: {prefixo_formatado}"
    else:
        return "Veiculo nao localizado na lista de aprovados."

def mostra_results(frame, boxes, labels, prefix, licenciado):
    #Coloca as boxes no frame congelado
    for box, cls in zip(boxes, labels):
        x1, y1, x2, y2, conf, cls = box
        x1 = max(0, int(x1 - 10))
        y1 = max(0, int(y1 - 10))
        x2 = min(frame.shape[1], int(x2 + 10))
        y2 = min(frame.shape[0], int(y2 + 10))

        cor = (0, 255, 0) 
        text = f"{licenciado}"

        if licenciado.startswith("Veiculo nao localizado"):
            cor = (0, 0, 255) 
            text = f"{licenciado}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

# Detecção em tempo real, verificação e congelamento
def main():
    cap = cv2.VideoCapture(1)
    frame_cng = None 
    congelado = False  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Verificar se a imagem está congelada
        if not congelado:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Detecção das vans
            ti = time.time()
            results_van = model_van(pil_image)
            boxes_van = results_van[0].boxes.data.cpu().numpy()
            labels_van = results_van[0].boxes.cls.cpu().numpy()

            for i, (box, cls) in enumerate(zip(boxes_van, labels_van)):
                if cls == 0:
                    x1, y1, x2, y2, conf, cls = box
                    x1 = max(0, x1 - 10)
                    y1 = max(0, y1 - 10)
                    x2 = min(pil_image.width, x2 + 10)
                    y2 = min(pil_image.height, y2 + 10)

                    van_crop = pil_image.crop((x1, y1, x2, y2))

                    prefix = ler_prefix(van_crop)
                    if prefix == "":
                        continue 

                    placa = ler_placa(van_crop)
                    licenciado = verifica_lista(prefix, placa)

                    # Desenhar caixas e texto no frame
                    mostra_results(frame, [box], [cls], prefix, licenciado)
                    tf = time.time()

                    print(f'Tempo de desempenho: {tf - ti}')

                    print(f"Prefixo: {prefix}, Placa: {placa}, Licenciado: {licenciado}")

                    # Congelar a imagem ao obter uma resposta
                    frame_cng = frame.copy()
                    congelado = True

        # Exibir a imagem congelada
        if congelado:
            cv2.imshow('Camera', frame_cng)
        else:
            cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)

        # Descongelar ao apertar espaço
        if key == ord(' ') and congelado:
            congelado = False

        #Encerra o processo ao apertar esc
        elif key == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()