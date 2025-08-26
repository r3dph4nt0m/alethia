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

class StructureMaoriDoc(SaveStructuredDialogue):
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, 'r', encoding='utf-8') as file:
            self.content = file.readlines()

        self.structured_maori = {
            "dialogue": [],
            "classification": [],
            "translation": []
        }
        self.save_file = "data/structured_maori.csv"
        super().__init__(self.structured_maori, self.save_file)

    def structure_maori(self):
        for line in self.content:
            if line.startswith(" ") or "Dialogue" in line:
                continue
            else:
                if line.startswith("T.") or line.startswith("P."):
                    filtered_line = line[2:].strip()
                    split_dialogue = filtered_line.split("  ")
                    self.structured_maori["dialogue"].append(split_dialogue[0].strip())
                    self.structured_maori["classification"].append("maori")
                    self.structured_maori["translation"].append(split_dialogue[len(split_dialogue) - 1].strip())

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
            if count >= 300:
                break




if __name__ == "__main__":
    maori_filepath = "data/docs_maori.txt"
    maori_doc = StructureMaoriDoc(maori_filepath)
    maori_doc.structure_maori()

    welsh_filepath = "data/docs_welsh.tsv"
    welsh_doc = StructureWelshDoc(welsh_filepath)
    welsh_doc.structure_welsh()

    ainu_filepath = "data/docs_ainu.json"
    ainu_doc = StructureAinuDoc(ainu_filepath)
    ainu_doc.structure_ainu()

    maori_doc.save()
    welsh_doc.save()
    ainu_doc.save()
