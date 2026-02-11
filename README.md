# Risk Engine

Modern medicine safety risk assessment engine built with LangChain + LangGraph.

## Project Overview
Risk Engine is designed to provide robust safety risk assessments for modern medicine. Using the power of LangChain and LangGraph, this system leverages advanced AI algorithms to analyze and respond to safety and risk evaluation tasks efficiently.

## Technical Overview / Architecture
This system adopts a modular architecture, ensuring scalability and maintainability. The components of the architecture are integrated seamlessly to ensure functionality and performance. For further details, you can explore the [DeepWiki documentation](https://deepwiki.com/MohammedNayeemE/risk-engine):

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MohammedNayeemE/risk-engine)

## Setup & Run Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your credentials

# Run the API
uvicorn app.main:app --reload
```

## Testing

```bash
pytest tests/
```

## Key Design Decisions
- **Framework Selection**: LangChain and LangGraph were chosen for their robust features, scalability, and community support.
- **Scalability**: Modular design to ensure components are loosely coupled and extensible.
- **Ease of Use**: Comprehensive setup instructions and environment configuration to support developers and contributors.
- **Testing Strategy**: Pytest used for reliable and efficient test management.

## Contributing
TBD