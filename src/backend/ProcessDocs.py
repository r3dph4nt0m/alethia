import pandas as pd
import os

class SaveStructuredDialogue:
    def __init__(self, structured_data, save_file):
        self.structured_data = structured_data
        self.save_file = save_file

    def save(self):
        df = pd.DataFrame(self.structured_data)
        if os.path.exists(self.save_file):
            df.to_csv(self.save_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.save_file, index=False)
        print(f"Saved {len(df)} dialogue pairs to {self.save_file}")

class StructureKhasiDoc(SaveStructuredDialogue):
    def __init__(self, filepath):
        self.filepath = filepath
        self.content = pd.read_csv(filepath, sep="\t", encoding='utf-8')

        self.structured_khasi = {
            "dialogue": [],
            "classification": [],
            "translation": []
        }
        self.save_file = "data/structured_khasi.csv"
        super().__init__(self.structured_khasi, self.save_file)

    def structure_khasi(self):
        for index, row in self.content.iterrows():
            self.structured_khasi["dialogue"].append(row['Text'])
            self.structured_khasi["classification"].append("khasi")
            self.structured_khasi["translation"].append(row["Translation"])

class StructureWelshDoc(SaveStructuredDialogue):
    def __init__(self, filepath):
        self.filepath = filepath
        self.content = pd.read_csv(filepath, sep='\t', encoding='utf-8')

        self.structured_welsh = {
            "dialogue": [],
            "classification": [],
            "translation": []
        }
        self.save_file = "data/structured_welsh.csv"
        super().__init__(self.structured_welsh, self.save_file)
    
    def structure_welsh(self):
        for index, row in self.content.iterrows():
            self.structured_welsh["dialogue"].append(row['Text'])
            self.structured_welsh["classification"].append("welsh")
            self.structured_welsh["translation"].append(row["Translation"])

class StructureAinuDoc(SaveStructuredDialogue):
    def __init__(self, filepath):
        self.filepath = filepath
        self.content = pd.read_json(filepath)
        self.save_file = "data/structured_ainu.csv"

        self.structured_ainu = {
            "dialogue": [],
            "classification": [],
            "translation": []
        }
        super().__init__(self.structured_ainu, self.save_file)

    def structure_ainu(self):
        count = 0
        for i, item in self.content.items():
            self.structured_ainu["dialogue"].append(item['transcription'])
            self.structured_ainu["classification"].append("ainu")
            self.structured_ainu["translation"].append(item["translation"])
            count += 1
            if count > 2000:
                break




if __name__ == "__main__":
    khasi_filepath = "data/docs_khasi.tsv"
    khasi_doc = StructureKhasiDoc(khasi_filepath)
    khasi_doc.structure_khasi()

    welsh_filepath = "data/docs_welsh.tsv"
    welsh_doc = StructureWelshDoc(welsh_filepath)
    welsh_doc.structure_welsh()

    ainu_filepath = "data/docs_ainu.json"
    ainu_doc = StructureAinuDoc(ainu_filepath)
    ainu_doc.structure_ainu()

    khasi_doc.save()
    welsh_doc.save()
    ainu_doc.save()
