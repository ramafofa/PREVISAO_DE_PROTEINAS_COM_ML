🧬 Sobre o projeto

Esse projeto nasceu do início dos meus estudos em biologia computacional, explorando livros e artigos sobre Biopython.
A ideia principal é criar um programa capaz de gerar uma sequência de aminoácidos (parte de uma proteína) a partir de um trecho de DNA fornecido pelo usuário.

O processo segue o mesmo princípio da expressão gênica:

Transcrição: o DNA é convertido em mRNA, substituindo as bases T (timina) por U (uracila).

Tradução: o RNA é dividido em trincas de bases (códons), e cada tríade corresponde a um aminoácido.

Síntese proteica: a combinação dos aminoácidos gera uma sequência proteica parcial — ou seja, uma parte da proteína original.

No exemplo usado no código, o programa reconstrói o início da sequência da Insulina, demonstrando o funcionamento completo da transcrição e tradução de uma sequência genética.

🤝 Colaboração e expansão do projeto

Depois de finalizar a primeira versão, compartilhei o projeto com a Ana Catarina, uma amiga engenheira de software.
Ela curtiu a ideia e resolveu expandir o projeto para a área de Machine Learning e, nesse processo, me ensinou conceitos de ML e também como criar gráficos para análise de dados.

Essa parceria acabou transformando o projeto em algo bem maior: agora ele não só traduz DNA em proteínas, como também abre caminho para análises preditivas e visualizações biológicas interativas,  gráficos que separam as variáveis a cerca dos aminoácidos, quais algorítimos utilizados para o ML etc.

(O trecho a seguir foi feito pela Ana, explicando o que está sendo feito).
📊 CONCEITOS BÁSICOS:

1. X (Features/Características) - "O QUE O MODELO VÊ"
   ─────────────────────────────────────────────────
   São as CARACTERÍSTICAS que usamos para descrever cada proteína.

   Exemplo prático:
   - Tamanho da proteína (150 aminoácidos)
   - % de aminoácidos hidrofóbicos (35%)
   - % de aminoácidos polares (20%)
   - % de aminoácidos carregados (15%)
   - % de aminoácidos aromáticos (10%)

   Pense assim: Se você quisesse descrever um cachorro para alguém que nunca viu,
   você diria: "4 patas, peludo, late, tamanho médio" → essas são as FEATURES!


2. y (Target/Alvo) - "O QUE QUEREMOS PREVER"
   ─────────────────────────────────────────
   É a RESPOSTA CORRETA que queremos que o modelo aprenda a prever.

   No nosso caso: O TIPO de proteína
   - hemoglobin (hemoglobina)
   - insulin (insulina)
   - myoglobin (mioglobina)
   - cytochrome (citocromo)

   Voltando ao exemplo: Se as features descrevem "4 patas, peludo, late" →
   o target (y) seria: "Cachorro"

═══════════════════════════════════════════════════════════════════════════════

🔄 PROCESSO DE TREINAMENTO:

PASSO 1: DIVIDIR OS DADOS
─────────────────────────

┌─────────────────────────────────────────┐
│  TODOS OS DADOS (42 proteínas)          │
└─────────────────────────────────────────┘
              │
              ▼
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────┐       ┌──────────┐
│ TREINO  │       │  TESTE   │
│  70%    │       │   30%    │
│(~29)    │       │  (~13)   │
└─────────┘       └──────────┘

Por que dividir?
- TREINO: O modelo APRENDE com esses dados (como estudar para prova)
- TESTE: Avaliamos se o modelo aprendeu bem (como fazer a prova de verdade)

Se testarmos com os mesmos dados do treino, é como dar a resposta da prova
para o aluno antes! Ele vai "decorar" mas não aprendeu de verdade.


PASSO 2: TREINAR O MODELO
──────────────────────────

O modelo (Regressão Logística) olha para os dados de TREINO e tenta encontrar
PADRÕES que conectam as características (X) com o tipo de proteína (y).

Exemplo do que o modelo aprende:
"Proteínas com tamanho pequeno (~50 aa) E alto % hidrofóbicos (~40%)
 geralmente são INSULINA"


PASSO 3: FAZER PREVISÕES
─────────────────────────

Depois de treinado, damos ao modelo uma proteína NOVA (do conjunto de teste)
e pedimos: "Que tipo de proteína é essa?"

Exemplo:
┌────────────────────────────────────────────┐
│ Proteína Desconhecida:                     │
│ - Tamanho: 51 aminoácidos                  │
│ - % hidrofóbicos: 38%                      │
│ - % polares: 18%                           │
│ - % carregados: 12%                        │
│ - % aromáticos: 8%                         │
└────────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────┐
    │  MODELO PENSA   │
    │  "Hmmm... isso  │
    │  parece com as  │
    │  insulinas que  │
    │  eu vi antes!"  │
    └─────────────────┘
              │
              ▼
    PREVISÃO: "INSULIN" ✓


PASSO 4: AVALIAR O MODELO
──────────────────────────

Comparamos as previsões do modelo com as respostas corretas (que já sabemos).

Exemplo:
┌──────────────┬────────────┬────────────┐
│ Proteína     │ Real       │ Previsto   │
├──────────────┼────────────┼────────────┤
│ Proteína 1   │ Insulin    │ Insulin ✓  │
│ Proteína 2   │ Hemoglobin │ Hemoglobin ✓│
│ Proteína 3   │ Myoglobin  │ Cytochrome ✗│
│ Proteína 4   │ Insulin    │ Insulin ✓  │
└──────────────┴────────────┴────────────┘

Acurácia = (Acertos / Total) × 100
         = (3 / 4) × 100 = 75%

═══════════════════════════════════════════════════════════════════════════════

📈 MÉTRICAS DE AVALIAÇÃO:

1. ACURÁCIA (Accuracy)
   ───────────────────
   Quantos % o modelo acertou no total?

   Se acertou 10 de 13 → Acurácia = 77%

2. PRECISÃO (Precision)
   ────────────────────
   "Quando o modelo diz que é Insulina, quantas vezes ele acerta?"

   Exemplo: Modelo disse "Insulina" 5 vezes → 4 eram Insulina de verdade
   Precisão = 4/5 = 80%

   Importante quando o ERRO é caro (ex: diagnóstico médico)


3. RECALL (Revocação)
   ──────────────────
   "De todas as Insulinas que existem, quantas o modelo conseguiu encontrar?"

   Exemplo: Existem 6 Insulinas no teste → modelo encontrou 4
   Recall = 4/6 = 67%

   Importante quando NÃO PODE PERDER nenhum caso (ex: detectar fraude)


4. MATRIZ DE CONFUSÃO
   ──────────────────
   Tabela que mostra onde o modelo acerta e onde erra:

                    PREVISTO
                 Ins  Hemo  Myo
   REAL    Ins  [ 4    1    0 ]  ← Das 5 Insulinas, acertou 4, errou 1
           Hemo [ 0    3    0 ]  ← Das 3 Hemoglobinas, acertou todas!
           Myo  [ 1    0    4 ]  ← Das 5 Mioglobinas, acertou 4, errou 1

   Diagonal = ACERTOS ✓
   Fora da diagonal = ERROS ✗

═══════════════════════════════════════════════════════════════════════════════

🎯 ANALOGIA COMPLETA:

Imagine que você quer ensinar alguém a identificar frutas:

1. X (Features): cor, tamanho, textura, cheiro
2. y (Target): tipo da fruta (maçã, banana, laranja)

3. Dados de TREINO: Você mostra 100 frutas e diz o nome de cada uma
   "Olha, isso é amarelo, comprido, macio → BANANA"
   "Olha, isso é vermelho, redondo, brilhante → MAÇÃ"

4. TREINAMENTO: A pessoa começa a perceber padrões
   "Ah, coisas amarelas e compridas costumam ser bananas!"

5. TESTE: Você mostra 30 frutas NOVAS (que ela nunca viu)
   Ela tenta adivinhar o nome

6. AVALIAÇÃO: Você conta quantas ela acertou
   Acertou 25 de 30 → 83% de acurácia!

7. PREVISÃO: Agora ela consegue identificar frutas novas sozinha
═══════════════════════════════════════════════════════════════════════════════
"""
(O trecho a seguir é um resumo baseado no que fiz e na avaliaçao da Ana)
*Resumo*:
O modelo tem 97.5% de acurácia, com o algorítimo SVM os resultados foram melhores.
🛠️TO DO:
Testar com outros algorítimos para que a máquina consiga prever perfeitamente a proteína. Ainda precisa-se de mais dados e mais testes. Mas, até o momento, como um mini-projeto em parceria com minha colega Ana Catarina, os resultados foram bons e aprendi bastante com ela!