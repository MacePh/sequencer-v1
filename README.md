# Learning Experience Agent Sequencer

A dynamic system that orchestrates personalized learning experiences for GenAI skills using a sequence of specialized learning agents.

## Overview

The Learning Experience Agent Sequencer analyzes a user's topic of interest and creates an optimal sequence of learning experiences tailored to that specific topic and the user's knowledge level. The system coordinates four specialized learning agents:

1. **Team Meeting Simulation Agent**: Simulates a workplace meeting between the user and 4 AI team members
2. **Socratic Learning Agent**: Engages critical thinking through targeted questioning
3. **Management Briefing Agent**: Simulates direct reports briefing the user on what they need to know
4. **Simulation Agent**: Provides a realistic workplace task with support from coach, assistant, and evaluator agents

## Architecture

The sequencer is built using the CrewAI framework, which allows seamless orchestration between the different agents. The system uses a Supabase database to store configurations, learning sequences, and engagement metrics.

## Features

- **Topic Analysis**: Automatically analyzes topic complexity to determine appropriate learning steps
- **Personalized Sequences**: Creates custom learning paths based on topic and user knowledge level
- **Engagement Metrics**: Tracks learning progress, interaction patterns, and completion rates
- **LLM Integration**: Supports multiple LLM providers (OpenAI, Anthropic Claude)
- **API-First Design**: Easily integrates with frontend applications

## Getting Started

### Prerequisites

- Python 3.8+
- Supabase account
- LLM API access (OpenAI API key or Anthropic API key)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-organization/learning-sequencer.git
cd learning-sequencer
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

Create a `.env` file based on the provided template:

```bash
cp .env.example .env
```

Edit the `.env` file with your Supabase and LLM API credentials.

### Database Setup

Run the database setup script:

```bash
python setup_database.py
```

This script will create all necessary tables in your Supabase project.

Note: Due to limitations in the Supabase Python client for executing raw SQL, you may need to run the SQL commands manually in the Supabase SQL Editor if the script encounters issues.

### Basic Usage

Run a test sequence generation:

```bash
python sample_run.py --topic "Prompt engineering for content marketing" --user_level intermediate
```

For more options:

```bash
python sample_run.py --help
```

## Key Components

### 1. Learning Sequencer

The main orchestrator that analyzes topics, determines optimal learning paths, and coordinates the learning agents. Found in `learning_sequencer.py`.

### 2. LLM Integration

Provides a flexible interface for connecting to different LLM providers (OpenAI, Anthropic Claude). Found in `llm_integration.py`.

### 3. Learning Agents

Specialized agents for different learning experiences:
- Team Meeting Agent: Simulates collaborative workplace discussions
- Socratic Learning Agent: Engages critical thinking through questions
- Management Briefing Agent: Provides executive-level insights
- Simulation Agent: Creates realistic workplace tasks

### 4. Database Schema

Stores configurations, learning sequences, and metrics for analysis. Key tables:
- `topic_analysis`: Stores analysis of learning topics
- `learning_sequences`: Stores the learning paths for users
- `agent_configurations`: Stores configurations for different learning agents
- `learning_metrics`: Tracks learning engagement and progress

## Development

### Debugging

For verbose logging and debugging, use the `--debug` flag:

```bash
python sample_run.py --topic "GenAI for content marketing" --debug --simulate_steps
```

### Testing without LLM

To test the system without connecting to an LLM API:

```bash
python sample_run.py --topic "Data analysis with GenAI" --llm_provider none
```

## API Integration

The system can be deployed as a REST API using FastAPI. See the deployment guide for details.

## Contributing

Contributions are welcome! Please check the [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
