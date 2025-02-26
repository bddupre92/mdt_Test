# MigraineDT: Migraine Prediction and Digital Twin Application

A comprehensive end-to-end migraine prediction application that leverages meta-optimization and concept drift detection to provide personalized migraine risk predictions.

## Features

- Real-time migraine risk prediction
- Multi-modal data integration
- Concept drift detection and adaptation
- Meta-optimizer framework for model selection
- Interactive visualization dashboard
- Patient diary and trigger tracking
- Researcher analytics interface

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **Frontend**: HTML5, CSS3, JavaScript (Chart.js)
- **Machine Learning**: NumPy, Pandas, Scikit-learn
- **Database**: PostgreSQL
- **Testing**: Pytest

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Initialize the database:
   ```bash
   python -m app.core.db.init_db
   ```

5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## Project Structure

```
app/
├── api/                 # API endpoints
│   ├── routes/         # Route definitions
│   └── dependencies.py # FastAPI dependencies
├── core/               # Business logic
│   ├── config/        # Configuration
│   ├── data/          # Data processing
│   ├── models/        # Database models
│   └── services/      # Business services
├── static/            # Static files
│   ├── css/          # Stylesheets
│   └── js/           # JavaScript
└── templates/         # HTML templates
    └── pages/        # Page templates
```

## API Documentation

Once the application is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Development

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run tests:
   ```bash
   pytest
   ```

3. Format code:
   ```bash
   black app
   flake8 app
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.