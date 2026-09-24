# Piloto Fake.Br: três LLMs e três temperaturas

> Documento histórico anterior à migração para Pixi. Para executar o projeto,
> consulte o README.md na raiz; os comandos e caminhos abaixo estão desatualizados.

## Primeiro: teste de viabilidade técnica

Antes de usar qualquer notícia da amostra, execute `viabilidade_deepinfra.py` com
`conf/viabilidade.yaml`. Este teste envia apenas textos sintéticos neutros. Verifica
se cada ID de modelo corresponde ao ID retornado, se a resposta é exatamente
VERDADEIRA ou FALSA, se há conteúdo de raciocínio, se `finish_reason=stop` e se a
resposta inclui dados de uso e custo. Não mede acurácia: os textos de controle
não têm rótulos de veracidade. Modelos atuais e preços podem mudar; confira a
disponibilidade na sua conta antes da execução.

Estrutura da pasta (os CSV são criados automaticamente):

```text
piloto-fakebr/
├── conf/
│   ├── config.yaml
│   └── viabilidade.yaml
├── fakebr_deepinfra_pilot.py
├── viabilidade_deepinfra.py
├── custos_experimentos.py             # sincroniza e resume o livro-caixa central
├── custos_totais.csv                  # criado automaticamente; não apague
├── test_fakebr_pilot.py
├── test_viabilidade.py
├── requirements_piloto.txt
├── README_piloto.md
├── Fake.br-Corpus/                   # após baixar o dataset; só usado no piloto
├── viabilidade_deepinfra/            # criado pelo teste técnico
│   ├── config.json
│   ├── results.csv
│   ├── costs.csv
│   └── summary.csv
└── piloto_multimodel/                # criado no prepare do piloto
```

Na raiz da pasta, após instalar as dependências e configurar `DEEPINFRA_TOKEN`,
execute (bash):

```bash
python -m pip install -r requirements_piloto.txt
export DEEPINFRA_TOKEN='SEU_TOKEN'
python -m unittest test_viabilidade test_fakebr_pilot -q
python viabilidade_deepinfra.py
```

São **15 chamadas planejadas**: cinco por modelo. Para cada modelo, há três
textos neutros de comprimentos diferentes na temperatura 0 e o texto curto nas
temperaturas 0,5 e 1. A primeira chamada de cada modelo é isolada; uma resposta
com modelo diferente, erro HTTP permanente ou falha técnica bloqueia as outras
quatro desse modelo nessa execução. O código usa `asyncio` e três requisições
concorrentes, com máximo de dois inícios por segundo. Até duas tentativas por
chamada são feitas somente para erros transitórios. Não há acesso ao Fake.Br
nesta etapa. Para investigar `logprobs`, execute um **novo teste com nova pasta**:

```bash
python viabilidade_deepinfra.py viability.out=viabilidade_com_logprobs viability.include_logprobs_probe=true
```

Isso acrescenta três chamadas, uma por modelo; apenas verifica se vieram
`top_logprobs`, sem calcular confiança nem presumir que os dois rótulos aparecem
entre os tokens retornados. A sonda adicional pode receber erro 400/422 e será
registrada como incompatibilidade da função, sem reprovar por si só a classificação.

Abra `viabilidade_deepinfra/summary.csv`: `base_approved=True` exige cinco
respostas válidas em cada modelo, ID correto, nenhum raciocínio no conteúdo,
`finish_reason=stop` e informações de uso. Examine também `results.csv` e
`costs.csv` para conferir o texto bruto, respostas inválidas, latências, tokens,
custos estimados e eventuais erros. Se `cost_source=unknown`, a conta pode ter
sido cobrada mesmo sem um valor disponível no CSV; confirme no extrato DeepInfra.
Um `False` é sinal para ajustar o modelo, a saída ou os parâmetros antes de
usar notícias reais; é um critério conservador deste teste, não uma medida de
qualidade científica.

Repetir `python viabilidade_deepinfra.py` com o mesmo YAML e pasta pula
requisições terminadas, inclusive respostas inválidas; tentativas que acabaram
em erro transitório são refeitas. `costs.csv` conserva todas as tentativas.
Se alterar modelos, temperatura, prompt, `max_tokens`, opções de cache ou a
sonda de `logprobs`, escolha outro `viability.out`: `config.json` e os hashes
impedem misturar configurações. O campo `prompt_cache_key` enviado à DeepInfra
identifica prefixos de prompt, **não** armazena respostas prontas. O controle
de retomada está no arquivo `results.csv`. Não inclua a chave de API nos arquivos.

Após aprovar o teste técnico, siga o procedimento abaixo para `action=prepare`
e `action=dev`. A amostra de desenvolvimento utiliza 20 notícias reais e tem
420 chamadas planejadas; as chamadas do teste técnico não entram nas métricas.

## Desenho registrado

Usamos o mesmo prompt A e a mesma amostra pareada do Fake.Br em todas as condições. A unidade de amostragem é o par Fake.Br: `dev_pairs: 10` dá **20 notícias** (10 VERDADEIRAS, 10 FALSAS); `dev_pairs: 5` dá **10 notícias**. O sorteio é fixado por `seed: 20260923`. As 400 notícias da avaliação (`eval_pairs: 200`) são sorteadas e reservadas desde o início, sem chamadas a elas até a decisão de expansão. Textos originais completos, zero-shot, sem busca, sem treinamento.

| Modelos | Temperatura | Repetições por notícia/modelo | Chamadas para 20 notícias |
| --- | ---: | ---: | ---: |
| DeepSeek V4.1 Flash, Qwen3.8 27B e Gemma 4 31B | 0 | 1 | 60 |
| Os mesmos três | 0,5 | 3 | 180 |
| Os mesmos três | 1 | 3 | 180 |
| **Total desenvolvimento** | | | **420** |

Com 10 notícias são 210 chamadas. Se a avaliação de 400 notícias for aprovada mais tarde, seu plano completo contém **8.400 chamadas** (400 × 3 modelos × 7 amostragens), além das tentativas adicionais em falhas. A etapa de desenvolvimento pode ser executada em duas partes: temperatura 0 (60 chamadas) e, somente após conferir as saídas, temperaturas 0,5 e 1 (360 chamadas). Temperatura 0 tem só uma repetição porque o objetivo das outras é estudar a variação causada pela amostragem. Com 20 notícias, as métricas são diagnósticas: não servem para escolher vencedor definitivo nem para estimar diferença pequena entre modelos.

Os preços por milhão de tokens no `conf/config.yaml` foram consultados em 23/09/2026 nas páginas oficiais [DeepSeek](https://deepinfra.com/deepseek-ai/DeepSeek-V4.1-Flash/api), [Qwen](https://deepinfra.com/Qwen/Qwen3.8-27B/api) e [Gemma](https://deepinfra.com/google/gemma-4-31B-it/api); valores promocionais e tarifas da sua conta podem variar. Exemplificando **1.000 tokens de entrada e 5 de saída por chamada, sem cache**, as 420 chamadas custariam aproximadamente US$ 0,061; a entrada efetiva do Fake.Br, eventual raciocínio e os tokens de saída podem alterar bastante essa estimativa. Compare depois `cost_summary.csv` com o extrato da conta.

## Preparar arquivos e dependências

Coloque `fakebr_deepinfra_pilot.py`, `README_piloto.md`, `requirements_piloto.txt` na mesma pasta e salve o YAML em `conf/config.yaml`. Em terminal nessa pasta:

```bash
python -m pip install -r requirements_piloto.txt
git clone https://github.com/roneysco/Fake.br-Corpus.git
export DEEPINFRA_TOKEN='SEU_TOKEN'
python fakebr_deepinfra_pilot.py action=prepare
```

No PowerShell, use `$env:DEEPINFRA_TOKEN="SEU_TOKEN"` no lugar de `export`. Para **10 notícias** em vez de 20, faça o `prepare` com `experiment.dev_pairs=5` antes de rodar. Para trocar pasta de dados/saída, edite o YAML antes de `prepare` ou use os mesmos overrides `experiment.dataset=... experiment.out=...` em todos os comandos. O script cria `resolved_config.yaml` e `config.json` com o plano e o hash da configuração. Não use a pasta de resultados do script anterior, que tinha um único modelo.

## Executar somente o desenvolvimento

Primeiro faça 60 chamadas determinísticas, distribuídas entre os três modelos:

```bash
python fakebr_deepinfra_pilot.py action=dev 'experiment.run_temperatures=[0.0]'
python fakebr_deepinfra_pilot.py action=report 'experiment.run_temperatures=[0.0]'
```

Confira `piloto_multimodel/responses.csv` (validade das respostas), `piloto_multimodel/costs.csv` (cada tentativa, tokens e custo) e `piloto_multimodel/report_dev/metrics.csv`. Se os três modelos responderem corretamente no formato e os custos estiverem dentro do esperado, complete as outras 360 chamadas:

```bash
python fakebr_deepinfra_pilot.py action=dev
python fakebr_deepinfra_pilot.py action=report
```

Os hashes permitem que o segundo `action=dev` pule chamadas já concluídas. O relatório completo produz `report_dev/metrics.csv` (cada repetição), `repeat_summary.csv` (média/desvio da macro-F1 e fração de notícias cuja classificação mudou entre as três repetições), `per_class.csv` e `confusion.csv`. As três repetições pertencem às **mesmas** 20 notícias: não aumentam o tamanho amostral para 60. Uma notícia cujo resultado alterna deve ser inspecionada junto com respostas inválidas.

Somente após decidir pela ampliação:

```bash
python fakebr_deepinfra_pilot.py action=eval
python fakebr_deepinfra_pilot.py action=report experiment.report_stage=eval
```

Isso gera `report_eval/`. Antes de executar `action=eval`, avalie o custo, tempo e qualidade do desenvolvimento; se desejar um desenho mais enxuto para a avaliação, congele outro YAML e outra pasta, pois mudar condições do experimento altera o hash registrado.

## Cache, taxa e limites

Cada chamada recebe `request_sha256` calculado sobre modelo, prompt completo, temperatura, parâmetros e índice da repetição. Assim a retomada local não duplica classificações concluídas, enquanto repetições estocásticas continuam independentes. `prompt_cache_key` é um hash **compartilhado por modelo, temperatura, repetição e prefixo**, enviado à DeepInfra; serve para o cache de prefixos e não guarda respostas. Veja `cached_tokens` antes de supor economia. O prefixo deste estudo é curto: provavelmente há pouco a reutilizar. Se algum modelo rejeitar o parâmetro, desligue `api.enable_prompt_cache_key` **em nova pasta** de experimento desde o `prepare`.

DeepSeek e Qwen têm campos de configuração para tentar desativar o modo de raciocínio, evitando que ele consuma os 16 tokens reservados à resposta; se a DeepInfra rejeitar algum campo para um modelo, a chamada inicial isolada dessa configuração interrompe a execução e exige ajuste explícito em um novo estudo. Não se solicita nem armazena cadeia de raciocínio. Como garantia de qualidade, examine `raw_response`, `prediction` e `finish_reason` do desenvolvimento; saída diferente de VERDADEIRA/FALSA é contabilizada como inválida.

O `api.target_rps` padrão é 75, `start_rps` é 5 e `api.concurrency` é 64. O programa aumenta gradualmente o limite de inícios por segundo e reduz após 429. Na conta padrão da DeepInfra há limite de 200 chamadas simultâneas **por modelo**, mas nem a taxa escolhida nem esse limite garantem 50–75 respostas por segundo: a latência e a capacidade disponível contam. Não há benefício em ajustar a taxa com base apenas nas 20 notícias se as chamadas já terminarem rapidamente.

`costs.csv` registra uma linha por tentativa, inclusive erros; `cost_summary.csv` agrupa etapa, modelo e temperatura. `provider_cost_usd` é o valor reportado pela API; `calculated_cost_usd` usa tarifas do YAML quando necessário; `accounted_cost_usd` privilegia o da API. Uma tentativa de erro sem dados de uso pode ter custo desconhecido, portanto o extrato é a referência de faturamento.

Além do arquivo local de cada execução, toda tentativa é acrescentada a
`custos_totais.csv` na raiz. O campo `ledger_id` evita dupla contagem nas
retomadas. Para importar custos de pastas já existentes e exibir os totais por
experimento e modelo, execute:

```bash
python custos_experimentos.py
```

Esse livro-caixa continua disponível se uma pasta individual de resultados for
apagada. Faça backup dele; o extrato da DeepInfra continua sendo a referência
definitiva quando uma tentativa não fornece dados de uso.

## Teste offline

`python -m unittest test_fakebr_pilot -q` verifica amostragem, SHA-256, retomada, três modelos, três temperaturas, repetições, custos e CSV, simulando respostas. Não consulta a DeepInfra.
