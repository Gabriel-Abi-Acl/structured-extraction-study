# Estudo de extração estruturada

Estudo técnico sobre extração de campos em textos curtos em português (pedidos, notas, chamados e endereços fictícios). O pacote deste PR deixa o contrato, o baseline e a medição prontos para rodar offline. Números de qualidade, latência de API e custo entram nos PRs seguintes, quando houver dataset rotulado e chamada real.

## Objetivo

Comparar três formas de preencher o mesmo registro:

1. Baseline determinístico de regras (`RegexExtractor`), que lê rótulos explícitos.
2. Extrator de LLM com validação JSON Schema (`LLMExtractor` com `max_attempts=1`).
3. O mesmo extrator com reparo limitado (`max_attempts` maior que 1): se a saída não valida, o erro do validador volta no diálogo, até o orçamento.

O estudo vizinho de guardrails trata estratégias de contenção. Aqui a pergunta é qualidade de extração contra o baseline de regras.

## Hipótese

Em texto curto e rotulado, o baseline deve acertar os campos com rótulo estável e perder quando o fraseado muda. O LLM com esquema deve cobrir melhor o texto livre. O reparo com orçamento deve recuperar JSON inválido ou fora do esquema, com um teto de chamadas. Um valor factualmente errado que ainda valida permanece erro de campo e não dispara nova tentativa.

Isso é o que o desenho permite medir. Este repositório ainda não traz essa medição.

## Métricas planejadas

| Métrica | Onde está | Leitura |
| --- | --- | --- |
| Correspondência exata por campo | `match_record` | Igualdade com o ouro, sem normalizar |
| Correspondência normalizada por campo | `match_record` | Os dois lados passam pelo normalizador do campo |
| Correspondência exata do registro | `RecordScore.exact` | Todos os campos do ouro fecham na forma exata |
| Taxa de JSON válido | `valid_json_rate` | Fração de saídas que `json.loads` aceita, depois de remover uma cerca markdown |
| Taxa de conformidade ao esquema | `schema_valid_rate` | Fração aceita pelo JSON Schema e pelo modelo Pydantic |
| Tentativas de reparo | `ExtractionResult.attempts` | Inclui a primeira chamada |
| Latência | `ExtractionResult.latency_ms` | Tempo medido dentro de `extract` |
| Custo | PR4 | Depende de uso real de tokens; não há tabela de preço neste PR |

Campo ausente no previsto não conta como nulo. Data impossível, valor ambíguo (`1.250` sem vírgula decimal) e identificador sem dígitos não são reescritos.

Os normalizadores cobrem data (`DD/MM/AAAA`, `DD-MM-AAAA`, `DD.MM.AAAA` e ISO), moeda em real, identificador fictício no formato de CPF (só dígitos, sem dígito verificador) e telefone (dígitos; o prefixo `55` sai quando o que resta tem tamanho nacional).

## Metodologia

- Os textos são sintéticos. Nomes são fictícios, e-mails usam `example.com`, identificadores imitam a máscara de CPF e não são documentos reais.
- O contrato do registro é `schemas/order.schema.json`, espelhado por `OrderRecord`. As chaves são obrigatórias; dado desconhecido é `null`.
- O baseline devolve a forma escrita no texto. A forma canônica fica na métrica, para a correspondência exata e a normalizada poderem divergir.
- O extrator de LLM recebe um cliente. Os testes usam `ScriptedLLMClient` (saída válida, JSON malformado, objeto fora do esquema, objeto válido com campo errado).
- Os SDKs `openai` e `anthropic` são extras opcionais e só são importados na primeira chamada ao vivo, depois da checagem de chave vazia.
- A fixture `fixtures/placeholder.json` tem 8 textos rotulados para travar o baseline. O dataset de avaliação é o PR2.

## Como rodar

Requer [uv](https://docs.astral.sh/uv/) e Python 3.11 ou superior.

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

A configuração vem do ambiente do processo. O pacote não lê arquivo `.env`. O exemplo versionado deixa todas as chaves vazias:

```bash
LLM_PROVIDER=
LLM_MODEL=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
REPAIR_MAX_ATTEMPTS=
```

Valor em branco usa o padrão: provedor `scripted`, orçamento `2`. `LLMExtractor` não lê o ambiente; quem for montar o experimento passa `load_settings().repair_max_attempts`. Cliente ao vivo, com extra e chave só na máquina local:

```bash
uv sync --extra openai
export LLM_PROVIDER=openai
export LLM_MODEL=<identificador do modelo>
export OPENAI_API_KEY=<chave local>
```

`build_live_client` monta o cliente e não chama a API. Chave real não entra no git.

O CI (GitHub Actions) instala uv com `astral-sh/setup-uv@v10.3.3`, Python 3.11, e roda ruff e pytest a partir do `uv.lock`. Os extras de provedor ficam de fora, então a suíte permanece offline.

## Roadmap

- PR2: dataset sintético rotulado, maior, com variação de fraseado.
- PR3: avaliação offline dos três extratores nesse dataset.
- PR4: experimentos com provedor real, latência, custo e relatório.

## Layout

```text
src/structured_extraction_study/
  schemas/          JSON Schema e modelo Pydantic
  extractors/       protocolo, baseline, extrator com reparo
  llm/              cliente scriptado e clientes ao vivo
  fixtures/         oito textos de exemplo
  normalize.py
  metrics.py
  config.py
```
