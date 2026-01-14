# Risk Engine

Modern medicine safety risk assessment engine built with LangChain + LangGraph.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for design principles and layered structure.

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your credentials

# Run the API
uvicorn app.main:app --reload
```

## API

- **POST /assess/start** – Start a risk assessment (returns thread ID if approval needed)
- **POST /assess/approve** – Resume a paused assessment with user input
- **GET /health** – Health check

## Testing

```bash
pytest tests/
```

## Contributing

TBD
