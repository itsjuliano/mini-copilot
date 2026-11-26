# Mini-Copilot

MiniGPT de nível de caractere (PyTorch, CPU) treinado para gerar código Java/Spring Boot, PostgreSQL e HTML5/Tailwind.

## Como reproduzir
1. `pip install torch --index-url https://download.pytorch.org/whl/cpu`
2. `pip install -r requirements.txt`
3. Clonar os repositórios de origem em `fontes/`
4. `python scripts/montar_dataset.py`
5. Executar `notebooks/minigpt.ipynb`


## Resultados

| Versão | Contexto | Passos | Validação | Melhor val loss |
|---|---|---|---|---|
| v1 | 128 | 8000 | 10% final do texto (arquivos misturados) | 0,483 |
| v2 | 128 | 6000 | repositórios separados (petclinic e windmill) | 1,309 |
| v3 | 256 | 4000 | repositórios separados (petclinic e windmill) | 1,313 |

- Modelo de cerca de 830 mil (v2) e 850 mil (v3) parâmetros, treinado em CPU
- A validação por repositório elevou a loss de 0,48 para 1,31, o que sugere que a validação inicial estava otimista (os conjuntos de validação são diferentes, então a comparação não é exata)
- Em v2 e v3, a loss de treino continua caindo (cerca de 0,55) enquanto a de validação estabiliza perto de 1,3 a partir de cerca de 3000 passos (sobreajuste); o checkpoint guardado é o de menor validação
- Dobrar o contexto (128 para 256) não alterou a validação, mas, nas poucas amostras geradas, a v3 produziu estruturas Java mais coerentes (getters e setters, chaves fechadas, fim de arquivo). É uma observação qualitativa, não medida
- O código gerado não compila

## Limitações
- Tokenização por caractere e contexto curto
- Poucos repositórios, com muitos arquivos parecidos entre si; o gargalo provável é a diversidade dos dados
- Pouco SQL no dataset (cerca de 0,4%)
- v2 e v3 diferem também no número de passos (6000 e 4000)