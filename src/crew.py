
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM for the agents
# Use environment variables for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4-turbo-preview")

# --- Define Tools ---
# Placeholder tools for demonstration. In a real-world scenario, these would
# perform actual searches, data analysis, or API calls.

@tool
def search_career_data(query: str) -> str:
    """Searches for general career information, job market trends, or educational paths."""
    # In a real implementation, this would call a web search API
    # or access a career database.
    return f"Simulated search results for '{query}': High demand for AI/ML engineers, growing tech sector, remote work opportunities increasing. Average salary data available for various roles."

@tool
def analyze_user_input(user_profile: str) -> str:
    """Analyzes the user's profile to extract key skills, interests, and aspirations."""
    # In a real implementation, this would parse a detailed text input
    # and structured data.
    return f"Simulated analysis of user profile: '{user_profile}'. Key skills identified: Python, Data Analysis, Communication. Interests: Technology, Problem Solving. Aspirations: Leadership role in tech."

# --- Define Agents ---

# Agent 1: Profile Analyst
profile_analyst = Agent(
    role='Profile Analyst',
    goal='Understand the user's background, skills, interests, and career aspirations.',
    backstory='An expert in human resources and career counseling, skilled at extracting core competencies and passions from user descriptions.',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[analyze_user_input]
)

# Agent 2: Market Research Agent
market_research_agent = Agent(
    role='Market Research Agent',
    goal='Provide up-to-date information on job market trends, in-demand skills, and industry insights.',
    backstory='A data-driven market researcher specializing in labor economics and technology sector analysis.',
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[search_career_data]
)

# --- Define Tasks ---

task1_analyze_profile = Task(
    description=(
        "Analyze the user's provided profile information: '{user_input}'"
        "Identify and list their core skills, interests, and long-term career aspirations."
        "The output should be a structured summary of the user's profile."
    ),
    expected_output='A structured summary of the user's profile including skills, interests, and aspirations.',
    agent=profile_analyst
)

task2_research_market = Task(
    description=(
        "Based on the analyzed user profile, research current job market trends, "
        "in-demand skills matching the user's profile, and relevant industry insights."
        "Focus on potential career paths that align with the user's characteristics."
        "The output should be a comprehensive report on market opportunities."
    ),
    expected_output='A comprehensive report detailing job market trends, in-demand skills, and aligning career paths.',
    agent=market_research_agent,
    context=[task1_analyze_profile]
)

# --- Assemble the Crew ---
career_crew = Crew(
    agents=[profile_analyst, market_research_agent],
    tasks=[task1_analyze_profile, task2_research_market],
    process=Process.sequential,
    verbose=True,
)

# --- Optional: Run the Crew (for testing) ---
def run_career_crew(user_input_data: str):
    print("## Career Orientation Crew Starting ##")
    inputs = {"user_input": user_input_data}
    result = career_crew.kickoff(inputs=inputs)
    print("

## Final Career Orientation Report ##")
    print(result)

if __name__ == "__main__":
    # Example usage:
    # Set your OPENAI_API_KEY in your .env file or as an environment variable
    # python src/crew.py
    user_data = "I am a recent computer science graduate with strong programming skills in Python and Java. I enjoy problem-solving and critical thinking. I'm looking for a career that allows me to work on innovative technologies and has good growth potential. I'm interested in roles related to AI, data science, or software development. I have some experience with machine learning projects from university."
    run_career_crew(user_data)
