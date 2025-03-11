import argparse
import json
from learning_sequencer import LearningSequencer
from llm_integration import create_llm_provider
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test the Learning Experience Sequencer")
    parser.add_argument("--topic", type=str, required=True, help="The GenAI topic to learn")
    parser.add_argument("--user_level", type=str, default="intermediate", choices=["beginner", "intermediate", "advanced"], help="User's knowledge level")
    parser.add_argument("--llm_provider", type=str, default=os.getenv("LLM_PROVIDER", "none"), choices=["openai", "anthropic", "none"], help="LLM provider to use")
    parser.add_argument("--simulate_steps", action="store_true", help="Simulate stepping through the learning experience")
    args = parser.parse_args()

    # Initialize LLM if specified
    llm = None
    if args.llm_provider.lower() != "none":
        try:
            llm = create_llm_provider(args.llm_provider)
            logger.info(f"LLM provider {args.llm_provider} initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM provider: {str(e)}")
            print("Continuing without LLM integration...")

    # Initialize sequencer
    sequencer = LearningSequencer(llm=llm)

    # Execute the sequence
    try:
        result = sequencer.execute_sequence(user_id="test-user", topic=args.topic, user_level=args.user_level)
        print(f"Sequence ID: {result['sequence_id']}")
        print("Learning Sequence Steps:")
        for step in result['steps']:
            agent_role = step['agent'].role
            focus = step['configuration'].get('focus', 'general')
            print(f"  - {agent_role}: {focus}")
        
        if args.simulate_steps:
            simulate_sequence(result['steps'])
    except Exception as e:
        logger.error(f"Error executing sequence: {str(e)}")
        print(f"Error: {str(e)}")

def simulate_sequence(steps):
    """Simulates stepping through the learning sequence."""
    for i, step in enumerate(steps):
        agent_role = step['agent'].role
        focus = step['configuration'].get('focus', 'general')
        print(f"\nStep {i+1}: {agent_role} - Focus: {focus}")
        input("Press Enter to continue to the next step...")

if __name__ == "__main__":
    main()