# Experimentos Fake.Br com DeepInfra

Experimentos reprodutíveis de classificação de veracidade com LLMs: vários
modelos, temperaturas, prompts e repetições. O passo a passo completo está em
`GUIA_EXECUCAO.txt`. Dependências e tarefas são geridas pelo [Pixi](https://pixi.sh/)
(Windows, Linux e macOS). Relatórios, testes, custos e preparação não chamam a API.

```text
conf/
  config.yaml              caminhos, API, corpora e recortes comuns a todos os experimentos
  experiment/<nome>.yaml   um arquivo por experimento (o nome usado na linha de comando)
  models/<nome>.yaml       um arquivo por modelo: id, extra_body e preços de referência
  prompts/<nome>.yaml      modelo de prompt, variantes (A, B, C) e rótulos
src/fakebr/
  cli.py                   entrada única das tarefas Pixi
  project.py               raiz do projeto, Hydra e configuração congelada dos runs
  api.py                   requisições, taxa adaptativa, novas tentativas e custos
  experiments/             executores: multimodel.py, viabilidade.py
  data.py metrics.py plots.py prompts.py cache.py costs.py io.py legacy.py
notebooks/resultados.ipynb relatório interativo (tabelas + figuras PNG/PDF)
tests/                     testes offline com cliente simulado
data/raw/Fake.br-Corpus/   corpus (pixi run fetch-data)
cache/                     respostas válidas em .pkl, por hash da requisição
runs/<run>/                resultados de cada execução + runs/custos_totais.csv
```

## Início rápido

```powershell
pixi install
pixi run test
pixi run fetch-data            # baixa o Fake.Br na revisão fixada (se data/ estiver vazio)
pixi run status                # lista experimentos e runs
```

Para chamadas novas, defina o token só na sessão: `$env:DEEPINFRA_TOKEN = 'seu-token'`
(veja no guia como fazer isso sem deixá-lo no histórico). As etapas abaixo podem
gerar custos:

```powershell
pixi run experiment viabilidade --run minha_viabilidade
pixi run experiment fakebr_multimodel --run meu_piloto --stage prepare
pixi run experiment fakebr_multimodel --run meu_piloto --stage dev
pixi run experiment fakebr_multimodel --run meu_piloto --stage eval
pixi run report fakebr_multimodel --run meu_piloto --stage eval
pixi run costs
```

## Relatórios e notebook

`pixi run report` grava tabelas e figuras na pasta do run (`report_dev/`,
`report_eval/` ou `report/`). Para PDF (vetorial, bom para a dissertação) ou
SVG, acrescente `"report.formats=[png,pdf]"`. As figuras não têm data embutida:
o mesmo run gera sempre os mesmos arquivos.

`notebooks/resultados.ipynb` faz o mesmo de forma interativa: escolha
experimento, run e etapa na célula de parâmetros, e ele gera o relatório em PNG
e PDF, mostra as tabelas e as figuras. Opcionalmente, também executa etapas antes
do relatório. Abra com `pixi run notebook` ou, no VS Code, selecione o kernel
`.pixi\envs\default`. Antes de versionar o notebook, limpe as saídas
(*Clear All Outputs*).

## Como a reprodutibilidade é garantida

- **Configuração congelada.** A primeira etapa de um run grava
  `runs/<run>/experiment.yaml` com modelos, prompts (texto completo), condições,
  amostra e URL da API. `dev`, `eval`, `report` e `cache-import` leem **só** esse
  arquivo. Editar `conf/` depois disso não muda nem quebra runs existentes.
- **Overrides ao retomar** são limitados ao ritmo da API (`api.concurrency`,
  `api.target_rps`...), a caminhos (`paths.*`) e a recortes (`select.*`, p.ex.
  `select.temperatures=[0.0]`). Qualquer outra mudança exige um run novo; as
  requisições idênticas vêm do cache, sem custo.
- **Corpus verificado.** `datasets.fakebr` em `conf/config.yaml` fixa a revisão
  git e a impressão digital dos textos. `prepare` recusa um corpus diferente, e
  cada etapa confere o SHA-256 de cada notícia. Nenhum caminho absoluto é
  gravado: o projeto pode ser movido ou copiado para outra máquina.
- **Proveniência.** `provenance.json` registra Python, pacotes, hash do código,
  commit git e hash do `pixi.lock`. `executions.jsonl` registra cada comando
  executado no run, com o hash do código daquele momento.
- **Compatibilidade.** `tests/test_compat.py` garante que prompts e chaves de
  requisição não mudem. Runs antigos (só com `config.json`) são convertidos
  automaticamente e verificados pelos hashes registrados.

## Adicionar um experimento

- **Mesma lógica, outra configuração** (modelos, prompts, temperaturas, amostra,
  `size_normalized_texts`...): copie `conf/experiment/fakebr_multimodel.yaml`
  para `conf/experiment/novo.yaml` e edite. Não é preciso mexer em código.
  `pixi run experiment novo --run ... --stage prepare` já funciona.
- **Novo modelo:** crie `conf/models/<nome>.yaml` e cite `<nome>` em `models:`.
- **Novo prompt:** crie `conf/prompts/<nome>_v2.yaml` (não edite um existente) e
  aponte `prompt_set:` para ele.
- **Lógica nova:** crie `src/fakebr/experiments/<executor>.py` com a interface
  descrita em `src/fakebr/experiments/__init__.py` e use `runner: <executor>` no
  YAML. A CLI encontra o módulo sozinha.

## Cache e segurança

O cache é `cache/responses/` (classificação) e `cache/viability/` (sondas), com
arquivos `<hash>.pkl`. Uma resposta com o mesmo hash é reaproveitada em outro
run, sem chamada nem novo lançamento de custo. Não abra `.pkl` de fontes não
confiáveis: pickle pode executar código durante a leitura. Faça backup de
`runs/`, `cache/`, `conf/`, `pyproject.toml` e `pixi.lock`; `data/` pode ser
baixado de novo com `pixi run fetch-data`.
