from crewai import Agent
import os
import supabase
from typing import List, Dict, Any, Optional
import json
import logging
from dotenv import load_dotenv
from crewai.tools import BaseTool  # Use CrewAI's BaseTool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define input schemas for tools
class AnalyzeTopicInput(BaseModel):
    topic: str = Field(..., description="The topic to analyze")

class DetermineSequenceInput(BaseModel):
    topic: str = Field(..., description="The topic to sequence")
    user_level: str = Field(default="intermediate", description="The user's knowledge level")

class StoreSequenceInput(BaseModel):
    user_id: str = Field(..., description="The user ID")
    topic: str = Field(..., description="The topic of the sequence")
    sequence: List[Dict[str, Any]] = Field(..., description="The learning sequence as a list of dictionaries")

class RetrieveConfigInput(BaseModel):
    agent_type: str = Field(..., description="The type of agent")
    topic: str = Field(..., description="The topic of the configuration")
    focus: str = Field(..., description="The focus area of the configuration")

# Define custom tool classes with schemas and Config
class AnalyzeTopicComplexityTool(BaseTool):
    name: str = "analyze_topic_complexity"
    description: str = "Analyzes the complexity and structure of a learning topic."
    sequencer: 'LearningSequencer'
    args_schema: type[BaseModel] = AnalyzeTopicInput
    
    class Config:
        arbitrary_types_allowed = True  # Allow LearningSequencer without full validation
    
    def __init__(self, sequencer):
        super().__init__(sequencer=sequencer)
        self.sequencer = sequencer
    
    def _run(self, topic: str) -> str:
        result = self.sequencer.analyze_topic_complexity(topic)
        return json.dumps(result)
    
    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")

class DetermineLearningSequenceTool(BaseTool):
    name: str = "determine_learning_sequence"
    description: str = "Determines the optimal sequence of learning experiences."
    sequencer: 'LearningSequencer'
    args_schema: type[BaseModel] = DetermineSequenceInput
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, sequencer):
        super().__init__(sequencer=sequencer)
        self.sequencer = sequencer
    
    def _run(self, topic: str, user_level: str = "intermediate") -> str:
        result = self.sequencer.determine_learning_sequence(topic, user_level)
        return json.dumps(result)
    
    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")

class StoreSequenceTool(BaseTool):
    name: str = "store_sequence"
    description: str = "Stores the learning sequence in the database."
    sequencer: 'LearningSequencer'
    args_schema: type[BaseModel] = StoreSequenceInput
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, sequencer):
        super().__init__(sequencer=sequencer)
        self.sequencer = sequencer
    
    def _run(self, user_id: str, topic: str, sequence: List[Dict[str, Any]]) -> str:
        result = self.sequencer.store_sequence(user_id, topic, sequence)
        return json.dumps({"sequence_id": result})
    
    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")

class RetrieveAgentConfigTool(BaseTool):
    name: str = "retrieve_agent_config"
    description: str = "Retrieves the specific configuration for a learning agent."
    sequencer: 'LearningSequencer'
    args_schema: type[BaseModel] = RetrieveConfigInput
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, sequencer):
        super().__init__(sequencer=sequencer)
        self.sequencer = sequencer
    
    def _run(self, agent_type: str, topic: str, focus: str) -> str:
        result = self.sequencer.retrieve_agent_config(agent_type, topic, focus)
        return json.dumps(result)
    
    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")

class LearningSequencer:
    def __init__(
        self,
        supabase_client=None,
        llm=None,
        agent_configs: Optional[Dict[str, Dict]] = None
    ):
        if supabase_client is None:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            if not supabase_url or not supabase_key:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables or .env file")
            self.supabase_client = supabase.create_client(supabase_url, supabase_key)
        else:
            self.supabase_client = supabase_client
            
        self.llm = llm
        self.agent_types = ["team_meeting", "socratic_learning", "management_briefing", "simulation"]
        self.learning_agents = {
            "team_meeting": self._create_team_meeting_agent(agent_configs),
            "socratic_learning": self._create_socratic_agent(agent_configs),
            "management_briefing": self._create_management_briefing_agent(agent_configs),
            "simulation": self._create_simulation_agent(agent_configs)
        }
        self.sequencer_agent = self._create_sequencer_agent()
        logger.info("Learning Sequencer initialized successfully")
    
    def _create_sequencer_agent(self):
        """Creates the main sequencer agent with properly implemented tools."""
        tools = [
            AnalyzeTopicComplexityTool(self),
            DetermineLearningSequenceTool(self),
            StoreSequenceTool(self),
            RetrieveAgentConfigTool(self)
        ]
        agent_kwargs = {
            "role": "Learning Experience Sequencer",
            "goal": "Design the optimal sequence of learning experiences for users to master GenAI skills",
            "backstory": """You are an advanced AI education specialist that analyzes learning topics
            and user characteristics to create the most effective learning sequence.
            You understand instructional design, adult learning theory, and GenAI skill development.""",
            "verbose": True,
            "allow_delegation": True,
            "tools": tools
        }
        if self.llm:
            agent_kwargs["llm"] = self.llm
        return Agent(**agent_kwargs)
    
    def _create_team_meeting_agent(self, agent_configs=None):
        config = (agent_configs or {}).get("team_meeting", {})
        return Agent(
            role=config.get("role", "Team Meeting Facilitator"),
            goal=config.get("goal", "Simulate a realistic team meeting exploring GenAI applications"),
            backstory=config.get("backstory", """You orchestrate dynamic team meetings where the user interacts
            with 4 simulated team members with different perspectives on GenAI applications."""),
            verbose=True
        )
    
    def _create_socratic_agent(self, agent_configs=None):
        config = (agent_configs or {}).get("socratic_learning", {})
        return Agent(
            role=config.get("role", "Socratic Learning Guide"),
            goal=config.get("goal", "Challenge user understanding through questioning"),
            backstory=config.get("backstory", """You use the Socratic method to help users discover insights
            about GenAI through progressive questioning."""),
            verbose=True
        )
    
    def _create_management_briefing_agent(self, agent_configs=None):
        config = (agent_configs or {}).get("management_briefing", {})
        return Agent(
            role=config.get("role", "Management Briefing Specialist"),
            goal=config.get("goal", "Provide executive-level insights on managing teams with GenAI"),
            backstory=config.get("backstory", """You simulate briefings from direct reports to the user,
            focusing on GenAI management needs."""),
            verbose=True
        )
    
    def _create_simulation_agent(self, agent_configs=None):
        config = (agent_configs or {}).get("simulation", {})
        return Agent(
            role=config.get("role", "Workplace Simulation Coordinator"),
            goal=config.get("goal", "Create realistic workplace scenarios to test GenAI skills"),
            backstory=config.get("backstory", """You design immersive simulations where users apply GenAI
            knowledge, supported by coach, assistant, and evaluator agents."""),
            verbose=True
        )
    
    def analyze_topic_complexity(self, topic: str) -> Dict[str, Any]:
        analysis = {
            "topic": topic,
            "complexity_level": "medium",
            "prerequisite_knowledge": ["basic GenAI concepts"],
            "practical_components": ["application", "evaluation"],
            "management_aspects": ["team coordination"],
            "estimated_learning_time": "4 hours"
        }
        self.supabase_client.table("topic_analysis").insert(analysis).execute()
        return analysis
    
    def determine_learning_sequence(self, topic: str, user_level: str = "intermediate") -> List[Dict[str, str]]:
        topic_analysis = self.analyze_topic_complexity(topic)
        if user_level == "beginner":
            sequence = [
                {"agent_type": "socratic_learning", "duration": 20, "focus": "fundamentals"},
                {"agent_type": "team_meeting", "duration": 30, "focus": "applications"},
                {"agent_type": "management_briefing", "duration": 15, "focus": "implementation"},
                {"agent_type": "simulation", "duration": 45, "focus": "practice"}
            ]
        elif user_level == "intermediate":
            sequence = [
                {"agent_type": "team_meeting", "duration": 30, "focus": "strategy"},
                {"agent_type": "socratic_learning", "duration": 20, "focus": "advanced concepts"},
                {"agent_type": "management_briefing", "duration": 15, "focus": "team capability"},
                {"agent_type": "simulation", "duration": 45, "focus": "complex scenario"}
            ]
        else:  # advanced
            sequence = [
                {"agent_type": "management_briefing", "duration": 15, "focus": "innovation"},
                {"agent_type": "team_meeting", "duration": 30, "focus": "integration"},
                {"agent_type": "simulation", "duration": 45, "focus": "optimization"},
                {"agent_type": "socratic_learning", "duration": 20, "focus": "future directions"}
            ]
        for step in sequence:
            step["topic"] = topic
        return sequence
    
    def store_sequence(self, user_id: str, topic: str, sequence: List[Dict[str, Any]]) -> str:
        sequence_data = {
            "user_id": user_id,
            "topic": topic,
            "sequence": json.dumps(sequence),
            "created_at": "NOW()",
            "completed_steps": 0,
            "total_steps": len(sequence)
        }
        result = self.supabase_client.table("learning_sequences").insert(sequence_data).execute()
        return result.data[0]["id"] if result.data else "sequence-creation-failed"
    
    def retrieve_agent_config(self, agent_type: str, topic: str, focus: str) -> Dict[str, Any]:
        query = self.supabase_client.table("agent_configurations")\
            .select("*")\
            .eq("agent_type", agent_type)\
            .eq("topic", topic)\
            .eq("focus", focus)\
            .execute()
        if query.data and len(query.data) > 0:
            return query.data[0]
        else:
            if agent_type == "team_meeting":
                config = {
                    "team_members": 4,
                    "roles": ["Technical Lead", "Product Manager", "Data Scientist", "Business Analyst"],
                    "time_limit": 30
                }
            elif agent_type == "socratic_learning":
                config = {"question_count": 8, "time_per_question": 2}
            elif agent_type == "management_briefing":
                config = {"briefing_length": 15}
            else:  # simulation
                config = {"time_limit": 45}
            config["focus"] = focus
            self.supabase_client.table("agent_configurations").insert({
                "agent_type": agent_type,
                "topic": topic,
                "focus": focus,
                "configuration": config
            }).execute()
            return config
    
    def execute_sequence(self, user_id: str, topic: str, user_level: str = "intermediate") -> Dict[str, Any]:
        sequence = self.determine_learning_sequence(topic, user_level)
        sequence_id = self.store_sequence(user_id, topic, sequence)
        enriched_sequence = []
        for step in sequence:
            agent_type = step["agent_type"]
            focus = step["focus"]
            config = self.retrieve_agent_config(agent_type, topic, focus)
            enriched_sequence.append({
                "sequence_step": len(enriched_sequence) + 1,
                "agent": self.learning_agents[agent_type],
                "configuration": config
            })
        return {
            "sequence_id": sequence_id,
            "user_id": user_id,
            "topic": topic,
            "user_level": user_level,
            "steps": enriched_sequence
        }