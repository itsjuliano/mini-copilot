# Mini-Copilot

MiniGPT de nível de caractere (PyTorch, CPU) treinado para gerar código Java/Spring Boot, PostgreSQL e HTML5/Tailwind.

## Como reproduzir
1. `pip install torch --index-url https://download.pytorch.org/whl/cpu`
2. `pip install -r requirements.txt`
3. Clonar os repositórios de origem em `fontes/`
4. `python scripts/montar_dataset.py`
5. Executar `notebooks/minigpt.ipynb`