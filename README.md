# Credit Classifier AI

Classificador de aprovação de crédito em Python. Treina um `RandomForestClassifier` (scikit-learn) com o histórico de solicitantes e devolve a decisão **aceito / negado** para novos pedidos.

## Destaques

- `Pipeline` do scikit-learn: o `OneHotEncoder` aprende só com o treino e trata categorias nunca vistas (`handle_unknown="ignore"`)
- Divisão treino/validação estratificada e reprodutível (`random_state` fixo)
- Camadas separadas: domínio (regra de aprovação), repositório (CSV), serviço (modelo) e caso de uso (orquestração)
- Dependências injetadas no caso de uso, o que permite testar sem arquivos reais
- Tipagem estrita (`mypy --strict`), lint com `ruff` e testes com `pytest`

## Arquitetura

```
main.py                  # composição das dependências
src/
├── config.py            # caminhos, colunas e parâmetros
├── domain/              # ApprovalRule: regra que rotula o histórico
├── repositories/        # CsvCreditRepository: leitura/escrita dos CSVs
├── services/            # CreditModel: pré-processamento + Random Forest
└── use_cases/           # ClassifyCreditUseCase: treina, valida e classifica
data/                    # training_data.csv, testing_data.csv e results/
tests/
```

## Como executar

```bash
pip install -r requirements.txt
python main.py
```

O resultado é salvo em `data/results/predictions_credit_result.csv`.

## Observação sobre os dados

O dataset de treino é sintético e não traz o desfecho real do crédito. Por isso o rótulo vem da `ApprovalRule` (renda > 3000 e histórico de crédito positivo), e o modelo aprende a reproduzir essa regra. O foco do projeto é a estrutura da solução, não a qualidade preditiva.

## Desenvolvimento

```bash
pip install -e ".[dev]"
ruff check .
mypy src tests
pytest
```
