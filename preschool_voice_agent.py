#!/usr/bin/env python3
"""
Preschool Reading AI - Chained Voice Agent Example
Based on OpenAI's Agents SDK documentation for chained voice agents
"""

import asyncio
import os
from typing import Dict, Any
import numpy as np
import sounddevice as sd

# Install required packages:
# pip install openai-agents 'openai-agents[voice]' sounddevice numpy pydantic

try:
    from agents import Agent, function_tool, Runner, trace
    from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
    from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline
    from pydantic import BaseModel
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install with: pip install openai-agents 'openai-agents[voice]' sounddevice numpy pydantic")
    exit(1)

# Set your OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable")
    exit(1)

# =============================================================================
# TOOLS FOR PRESCHOOL READING
# =============================================================================

class ReadingProgress(BaseModel):
    """Model for tracking reading progress"""
    child_name: str
    level: str
    words_learned: int
    books_completed: int

@function_tool
def get_reading_progress(child_name: str) -> Dict[str, Any]:
    """Get a child's reading progress and achievements"""
    # Simulated progress data
    progress_data = {
        "Emma": {"level": "Beginning Reader", "words_learned": 45, "books_completed": 8},
        "Liam": {"level": "Early Reader", "words_learned": 78, "books_completed": 15},
        "Sophia": {"level": "Developing Reader", "words_learned": 120, "books_completed": 22}
    }
    
    if child_name in progress_data:
        return {
            "child_name": child_name,
            "level": progress_data[child_name]["level"],
            "words_learned": progress_data[child_name]["words_learned"],
            "books_completed": progress_data[child_name]["books_completed"],
            "next_milestone": "Great job! Keep reading to reach the next level!"
        }
    else:
        return {
            "child_name": child_name,
            "level": "New Student",
            "words_learned": 0,
            "books_completed": 0,
            "next_milestone": "Welcome! Let's start your reading journey!"
        }

@function_tool
def get_sight_words(difficulty_level: str) -> Dict[str, Any]:
    """Get sight words for different difficulty levels"""
    sight_words = {
        "beginner": ["the", "and", "a", "to", "said", "you", "of", "we", "my", "be"],
        "intermediate": ["have", "from", "they", "know", "want", "been", "good", "much", "some", "time"],
        "advanced": ["would", "there", "each", "which", "their", "called", "first", "water", "after", "back"]
    }
    
    level = difficulty_level.lower()
    if level in sight_words:
        return {
            "level": level,
            "words": sight_words[level],
            "practice_tip": f"Practice these {level} sight words by saying them out loud!"
        }
    else:
        return {
            "level": "beginner",
            "words": sight_words["beginner"],
            "practice_tip": "Let's start with beginner words!"
        }

@function_tool
def create_phonics_exercise(letter_sound: str) -> Dict[str, Any]:
    """Create a phonics exercise for a specific letter sound"""
    phonics_exercises = {
        "b": {"words": ["ball", "bear", "book", "banana"], "sentence": "Big brown bears bounce balls."},
        "c": {"words": ["cat", "car", "cake", "cup"], "sentence": "Curious cats catch colorful cars."},
        "d": {"words": ["dog", "duck", "door", "dance"], "sentence": "Dancing dogs dive through doors."},
        "f": {"words": ["fish", "frog", "flower", "family"], "sentence": "Funny frogs find fresh flowers."},
        "m": {"words": ["mouse", "moon", "milk", "music"], "sentence": "Mice make merry music under the moon."}
    }
    
    sound = letter_sound.lower()
    if sound in phonics_exercises:
        return {
            "letter_sound": sound.upper(),
            "practice_words": phonics_exercises[sound]["words"],
            "practice_sentence": phonics_exercises[sound]["sentence"],
            "instruction": f"Let's practice the '{sound}' sound together! Repeat after me."
        }
    else:
        return {
            "letter_sound": "B",
            "practice_words": ["ball", "bear", "book", "banana"],
            "practice_sentence": "Big brown bears bounce balls.",
            "instruction": "Let's practice the 'B' sound together! Repeat after me."
        }

# =============================================================================
# SPECIALIZED AGENTS FOR PRESCHOOL READING
# =============================================================================

# Patient Teacher System Prompt - Based on OpenAI's guidance
patient_teacher_prompt = """
You are a warm, patient, and encouraging preschool reading teacher. Your voice will be converted to speech, so:

PERSONALITY:
- Speak with enthusiasm and warmth
- Use simple, clear language appropriate for ages 3-6
- Be patient and encouraging, celebrating small wins
- Use a gentle, nurturing tone that builds confidence

TEACHING APPROACH:
- Break down complex concepts into simple steps
- Use repetition and reinforcement naturally
- Ask engaging questions to keep children involved
- Provide specific praise and encouragement
- Make learning fun and interactive

VOICE DELIVERY:
- Speak slowly and clearly for young learners
- Use varied intonation to maintain engagement
- Pause between instructions to allow processing time
- Use excitement in your voice for achievements
- Keep responses brief but complete (1-2 sentences maximum)
"""

# 1. PHONICS SPECIALIST AGENT
phonics_agent = Agent(
    name="PhonicsTeacher",
    handoff_description="Specialist for letter sounds, phonics, and pronunciation practice",
    instructions=patient_teacher_prompt + """
    
    You are the phonics specialist! You help children learn:
    - Letter sounds and pronunciation
    - Phonics exercises and games
    - Sound blending activities
    - Letter recognition practice
    
    Use the create_phonics_exercise tool to provide targeted practice.
    Always make phonics fun and interactive!
    """,
    tools=[create_phonics_exercise]
)

# 2. SIGHT WORDS SPECIALIST AGENT  
sight_words_agent = Agent(
    name="SightWordsTeacher",
    handoff_description="Specialist for sight words and high-frequency word recognition",
    instructions=patient_teacher_prompt + """
    
    You are the sight words specialist! You help children learn:
    - Common sight words and high-frequency words
    - Word recognition practice
    - Reading fluency through sight word mastery
    - Memory techniques for word retention
    
    Use the get_sight_words tool to provide appropriate word lists.
    Make sight word practice engaging and memorable!
    """,
    tools=[get_sight_words]
)

# 3. PROGRESS TRACKER AGENT
progress_agent = Agent(
    name="ProgressTracker", 
    handoff_description="Specialist for tracking reading progress and celebrating achievements",
    instructions=patient_teacher_prompt + """
    
    You are the progress specialist! You help by:
    - Tracking each child's reading journey
    - Celebrating milestones and achievements
    - Providing personalized encouragement
    - Setting appropriate next goals
    
    Use the get_reading_progress tool to check a child's current level.
    Always celebrate progress and motivate continued learning!
    """,
    tools=[get_reading_progress]
)

# 4. MAIN TRIAGE AGENT - Routes to appropriate specialist
main_teacher_agent = Agent(
    name="MainTeacher",
    instructions=prompt_with_handoff_instructions(f"""
    {patient_teacher_prompt}
    
    You are the main preschool reading teacher! You greet children warmly and determine how to help them today.
    
    Based on what the child needs, route them to the right specialist:
    - PhonicsTeacher: for letter sounds, phonics, and pronunciation
    - SightWordsTeacher: for sight words and word recognition  
    - ProgressTracker: for checking progress and celebrating achievements
    
    Always start with a warm greeting and ask how you can help them learn to read today!
    """),
    handoffs=[phonics_agent, sight_words_agent, progress_agent]
)

# =============================================================================
# VOICE PIPELINE CONFIGURATION
# =============================================================================

async def run_text_example():
    """Run a text-based example to show the chained agents in action"""
    print("=== PRESCHOOL READING AI - TEXT EXAMPLE ===\n")
    
    test_queries = [
        "Hi! I want to practice the letter B sound",
        "Can you check Emma's reading progress?", 
        "I need help with sight words for beginners",
        "Let's work on phonics with the letter M"
    ]
    
    with trace("Preschool Reading Text Demo"):
        for query in test_queries:
            print(f"👶 Child: {query}")
            result = await Runner.run(main_teacher_agent, query)
            print(f"👩‍🏫 Teacher: {result.final_output}")
            print("-" * 50)

async def run_voice_example():
    """Run the voice-based chained agent system"""
    print("\n=== PRESCHOOL READING AI - VOICE EXAMPLE ===")
    print("This will use your microphone and speakers!")
    print("Press Enter to speak, then press Enter again to stop recording.")
    print("Type 'quit' to exit.\n")
    
    # Get audio device info
    try:
        samplerate = int(sd.query_devices(kind='input')['default_samplerate'])
    except:
        samplerate = 44100  # fallback
        
    while True:
        # Get user input
        user_input = input("Press Enter to speak to the teacher (or type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print("Goodbye! Keep practicing your reading! 👋")
            break
            
        print("🎤 Listening... (Press Enter when done speaking)")
        
        # Record audio
        recorded_chunks = []
        def audio_callback(indata, frames, time, status):
            recorded_chunks.append(indata.copy())
            
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', callback=audio_callback):
            input()  # Wait for Enter key
            
        if not recorded_chunks:
            print("No audio recorded. Please try again.")
            continue
            
        # Process the recording
        print("🤔 Teacher is thinking...")
        
        # Create voice pipeline
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(main_teacher_agent))
        
        # Concatenate recorded audio
        recording = np.concatenate(recorded_chunks, axis=0)
        audio_input = AudioInput(buffer=recording)
        
        try:
            # Run the voice pipeline with tracing
            with trace("Preschool Reading Voice Demo"):
                result = await pipeline.run(audio_input)
                
                # Collect response audio
                response_chunks = []
                async for event in result.stream():
                    if event.type == "voice_stream_event_audio":
                        response_chunks.append(event.data)
                
                if response_chunks:
                    # Play the response
                    print("👩‍🏫 Teacher is responding...")
                    response_audio = np.concatenate(response_chunks, axis=0)
                    sd.play(response_audio, samplerate=samplerate)
                    sd.wait()  # Wait for playback to finish
                else:
                    print("No audio response generated.")
                    
        except Exception as e:
            print(f"Error processing voice: {e}")
            
        print("=" * 50)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main function to demonstrate the chained voice agent system"""
    print("🎓 PRESCHOOL READING AI - CHAINED VOICE AGENT DEMO")
    print("Based on OpenAI's Agents SDK with Patient Teacher Instructions")
    print("=" * 60)
    
    # First run text examples to show the chained agents
    await run_text_example()
    
    # Ask user if they want to try voice mode
    voice_choice = input("\nWould you like to try voice mode? (y/n): ").lower()
    if voice_choice == 'y':
        await run_voice_example()
    
    print("\n✨ Thank you for trying the Preschool Reading AI!")
    print("This demo shows how chained agents can work together")
    print("to provide specialized educational support through voice.")

if __name__ == "__main__":
    asyncio.run(main()) 