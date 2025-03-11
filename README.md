# Learning Experience Agent Sequencer

A system that orchestrates personalized learning experiences for GenAI skills using a sequence of specialized learning agents.

## Overview

The Learning Experience Agent Sequencer analyzes a user's topic of interest and creates an optimal sequence of learning experiences tailored to that specific topic and the user's knowledge level. The system coordinates four specialized learning agents:

1. **Team Meeting Simulation Agent**: Simulates a workplace meeting between the user and 4 AI team members
2. **Socratic Learning Agent**: Engages critical thinking through targeted questioning
3. **Management Briefing Agent**: Simulates direct reports briefing the user on what they need to know
4. **Simulation Agent**: Provides a realistic workplace task with support from coach, assistant, and evaluator agents

## Architecture

The sequencer is built using the CrewAI framework, which allows seamless orchestration between the different agents. The system uses a Supabase database to store configurations, learning sequences, and engagement metrics.

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
# sequencer-v1
