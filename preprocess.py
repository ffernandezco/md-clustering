import csv


class Instance:
    def __init__(self, pcm, rs, pu, txt):
        self.pcm = pcm
        self.rs = rs
        self.pu = pu
        self.txt = txt

    def getPCM(self):
        return self.pcm

    def getRS(self):
        return self.rs

    def getPU(self):
        return self.pu

    def getTXT(self):
        return self.txt

    def setPCM(self):
        self.pcm = 1

    def setPU(self):
        self.pu = 1


def parse_int_or_zero(value):
    try:
        return int(value)
    except ValueError:
        return 0


def preprocess(input_file, output_file):
    # Leer el archivo CSV
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    split_rows = []

    for row in rows:
        # Unir los elementos de la fila en un solo string
        joined_row = ",".join(row)

        # Dividir la fila en exactamente 5 partes
        split = joined_row.split(",", 4)
        split_rows.append(split)

    rows = split_rows

    # Eliminar la primera fila (cabecera)
    if rows:
        rows.pop(0)

    # Estructura de datos
    data = {}

    for row in rows:
        ui = row[0]
        pcm = parse_int_or_zero(row[1])
        rs = parse_int_or_zero(row[2])
        pu = parse_int_or_zero(row[3])
        txt = row[4].replace('|', '')  # Eliminar el carácter "|"
        instance = Instance(pcm, rs, pu, txt)

        if ui in data:
            data[ui].append(instance)
        else:
            data[ui] = [instance]

    # Borrar instancias falsas
    if "0" in data:
        del data["0"]

    print("Archivo CSV importado correctamente.")

    for user, instances in data.items():
        set_pcm = False
        set_pu = False

        for i in instances:
            if i.getPCM() == 1:
                set_pcm = True
            if i.getPU() == 1:
                set_pu = True
                break

        if set_pcm and set_pu:
            for i in instances:
                i.setPCM()
                i.setPU()

    print("Preprocess efectuado correctamente.")

    # Escribir el archivo de salida
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')

        for user, instances in data.items():
            for i in instances:
                new_row = [user, str(i.getPCM()), str(i.getRS()), str(i.getPU()), i.getTXT()]
                writer.writerow(new_row)

    print("Archivo postprocesado generado correctamente.")
