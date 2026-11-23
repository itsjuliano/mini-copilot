import hashlib
import random
import re
from pathlib import Path

# Pasta raiz do projeto (a pasta acima de "scripts")
BASE = Path(__file__).resolve().parent.parent
FONTES = BASE / "fontes"
SAIDA = BASE / "dados" / "dataset.txt"

EXTENSOES = {".java", ".sql", ".html"}
IGNORAR = {".git", "target", "build", "node_modules", ".idea", ".mvn", "out", "dist"}
MAX_BYTES = 60_000  # ignora arquivos muito grandes


def limpar(txt: str) -> str:
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    txt = txt.replace("\t", "    ")
    txt = "\n".join(linha.rstrip() for linha in txt.split("\n"))
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip() + "\n"


arquivos = []
vistos = set()
por_extensao = {}

for repo in sorted(p for p in FONTES.iterdir() if p.is_dir()):
    for caminho in repo.rglob("*"):
        if not caminho.is_file():
            continue
        if caminho.suffix.lower() not in EXTENSOES:
            continue
        if any(parte in IGNORAR for parte in caminho.parts):
            continue
        if caminho.stat().st_size > MAX_BYTES:
            continue

        try:
            conteudo = limpar(caminho.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue

        normalizado = re.sub(r"\d+", "0", conteudo)  # Customer1449 e Customer1454 viram o mesmo texto
        h = hashlib.md5(normalizado.encode("utf-8")).hexdigest()
        if h in vistos:  # remove duplicados
            continue
        vistos.add(h)

        nome = f"{repo.name}/{caminho.relative_to(repo).as_posix()}"
        arquivos.append((nome, conteudo))

        ext = caminho.suffix.lower()
        por_extensao[ext] = por_extensao.get(ext, 0) + 1

# Orçamento máximo de caracteres por tipo (ajuste depois de ver o resultado)
ORCAMENTO = {".java": 2_000_000, ".html": 1_500_000, ".sql": 500_000}

random.seed(42)
random.shuffle(arquivos)

usado = {}
selecionados = []
for nome, conteudo in arquivos:
    ext = Path(nome).suffix.lower()
    if usado.get(ext, 0) + len(conteudo) > ORCAMENTO[ext]:
        continue
    usado[ext] = usado.get(ext, 0) + len(conteudo)
    selecionados.append((nome, conteudo))

SAIDA.parent.mkdir(exist_ok=True)
with open(SAIDA, "w", encoding="utf-8") as f:
    for nome, conteudo in selecionados:
        f.write(f"@@@ ARQUIVO: {nome} @@@\n")
        f.write(conteudo)
        f.write("@@@ FIM @@@\n\n")

print("Arquivos selecionados:", len(selecionados))
print("Caracteres por extensão:", usado)
print(f"Tamanho do dataset: {SAIDA.stat().st_size / 1_000_000:.2f} MB")
print("Salvo em:", SAIDA)