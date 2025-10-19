# -*- coding: utf-8 -*-
import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings('ignore', category=BiopythonDeprecationWarning)

# Imports principais
from Bio import pairwise2
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo
from Bio.Seq import Seq
from Bio.Data import CodonTable
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.exceptions import DataConversionWarning
from Bio import Entrez, SeqIO
Entrez.email = "ramaechelsea@gmail.com"
Entrez.email = "ancoaraujo@gmail.com"

warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ====================================================
# PARTE 1 – TRADUÇÃO DNA → PROTEÍNA
# ====================================================

sequencia_de_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
sequencia_de_dna2 = Seq("CAAAGATACCAGGTCCCCAACAACGCAACTTTCTGGGA")

mRNA = sequencia_de_dna.transcribe()
print("mRNA:", mRNA)

proteina_completa = mRNA.translate(to_stop=False, cds=False)
print("Tradução completa:", proteina_completa)

tabela = CodonTable.unambiguous_dna_by_name["Standard"]

nomes_aminoacidos = {
    'A':'Alanina','R':'Arginina','N':'Asparagina','D':'Ácido Aspártico',
    'C':'Cisteína','E':'Ácido Glutâmico','Q':'Glutamina','G':'Glicina',
    'H':'Histidina','I':'Isoleucina','L':'Leucina','K':'Lisina',
    'M':'Metionina','F':'Fenilalanina','P':'Prolina','S':'Serina',
    'T':'Treonina','W':'Triptofano','Y':'Tirosina','V':'Valina','STOP':'STOP'
}

codons = [sequencia_de_dna[i:i+3] for i in range(0, len(sequencia_de_dna), 3)]
amino_acidos, nomes_completos = [], []

for codon in codons:
    if codon in tabela.stop_codons:
        amino_acidos.append("STOP")
        nomes_completos.append("STOP")
    else:
        aa = tabela.forward_table.get(str(codon), "X")
        amino_acidos.append(aa)
        nomes_completos.append(nomes_aminoacidos.get(aa, "Desconhecido"))

cell_colors = ['red' if aa=="STOP" else 'green' for aa in amino_acidos]
table_data = [codons, amino_acidos, nomes_completos]

fig, ax = plt.subplots(figsize=(14,3))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=table_data, cellColours=[cell_colors]*3,
         rowLabels=['Códon', 'Aminoácido', 'Nome Completo'], loc='center')
plt.title("Tradução DNA → Aminoácidos (Códon por Códon)")
plt.show()

proteina_ate_stop = mRNA.translate(to_stop=True, cds=False)
print("Tradução até STOP:", proteina_ate_stop)

contagem_aa = Counter(str(proteina_completa))
plt.figure(figsize=(8, 6))
plt.pie(contagem_aa.values(), labels=contagem_aa.keys(), autopct='%1.1f%%')
plt.title('Composição de Aminoácidos na Proteína')
plt.show()

# ======================
# SIMILARIDADE ENTRE SEQUÊNCIAS
# ======================
correspondencias = sum(1 for a, b in zip(str(sequencia_de_dna)[:len(sequencia_de_dna2)],
                                          str(sequencia_de_dna2)) if a == b)
diferencas = len(sequencia_de_dna2) - correspondencias
percentual_similaridade = (correspondencias / len(sequencia_de_dna2)) * 100

plt.figure(figsize=(5, 4))
plt.bar(['Correspondências', 'Diferenças'], [correspondencias, diferencas], color=['green', 'red'])
plt.ylabel('Número de bases')
plt.title(f'Similaridade entre Insulina de um Humano e um Rato (39 bases)\n({percentual_similaridade:.1f}% similar)')
plt.show()

print(f"Correspondências: {correspondencias}/{len(sequencia_de_dna2)}")
print(f"Diferenças: {diferencas}/{len(sequencia_de_dna2)}")
print(f"Similaridade: {percentual_similaridade:.1f}%")

##TO-FIX: NO COLLAB A ANA FEZ VÁRIOS GRÁFICOS EM RELAÇAO A TAMNHOS,TIPOS E PORCENTAGENS DOS AMINOÁCIDOS, MAS NAO ESTOU CONSEGUINDO COLOCAR NO MOMENTO. ERRO DE DATAFRAME


# ====================================================
# PARTE 2 – BUSCAR PROTEÍNAS REAIS (UniProt)
# ====================================================

def buscar_uniprot(protein_name, max_results=50):
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        'query': f'protein_name:{protein_name} AND reviewed:true',
        'format': 'tsv',
        'fields': 'accession,protein_name,organism_name,sequence,length',
        'size': max_results
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = []
            lines = response.text.strip().split('\n')
            for line in lines[1:]:
                cols = line.split('\t')
                if len(cols) >= 5:
                    data.append({
                        'id': cols[0],
                        'nome': cols[1],
                        'organismo': cols[2],
                        'sequencia_proteina': cols[3],
                        'tamanho_proteina': int(cols[4])
                    })
            return data
        return []
    except Exception as e:
        print(f"Erro ao buscar {protein_name}: {e}")
        return []

proteinas_buscar = [('hemoglobin',150),('insulin',150),('myoglobin',150),('cytochrome c',150)]
dados_proteinas = []

print("Buscando dados do UniProt...")
for nome, limite in proteinas_buscar:
    print(f"- {nome}...")
    resultados = buscar_uniprot(nome, limite)
    dados_proteinas.extend(resultados)
    print(f"  ✓ {len(resultados)} resultados")

df_original = pd.DataFrame(dados_proteinas)
print(f"\nTotal coletado: {len(df_original)} proteínas")

# ====================================================
# PARTE 3 – FEATURE ENGINEERING
# ====================================================

df_original['tipo_proteina'] = df_original['nome'].str.lower().str.extract(
    r'(hemoglobin|insulin|myoglobin|cytochrome)', expand=False
).fillna('outro')
df_original = df_original[df_original['tipo_proteina'] != 'outro'].reset_index(drop=True)

def calcular_features_proteina(seq):
    seq = str(seq)
    c = Counter(seq)
    total = len(seq)
    return {
        'tamanho': total,
        'pct_aromaticos': (c.get('F',0)+c.get('W',0)+c.get('Y',0))/total*100,
        'pct_polares': (c.get('S',0)+c.get('T',0)+c.get('N',0)+c.get('Q',0))/total*100,
        'pct_carregados': (c.get('K',0)+c.get('R',0)+c.get('D',0)+c.get('E',0))/total*100,
        'pct_hidrofobicos': (c.get('A',0)+c.get('V',0)+c.get('I',0)+c.get('L',0)+c.get('M',0))/total*100
    }

features = df_original['sequencia_proteina'].apply(calcular_features_proteina)
df_features = pd.DataFrame(features.tolist())
df = pd.concat([df_original, df_features], axis=1)

# ====================================================
# PARTE 4 – CLASSIFICAÇÃO (TODOS OS MODELOS)
# ====================================================

X = df[['tamanho','pct_hidrofobicos','pct_polares','pct_carregados','pct_aromaticos']]
y = df['tipo_proteina']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Lista de modelos
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
for name, model in models.items():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = accuracy_score(y_test, y_pred)

# Plotando acurácias
plt.figure(figsize=(10,6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylabel("Acurácia")
plt.ylim(0, 1.05)
plt.title("Comparação de modelos de classificação")
plt.xticks(rotation=45)
plt.show()
