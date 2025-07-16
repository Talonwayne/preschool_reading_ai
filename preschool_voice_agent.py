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
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install with: pip install openai-agents 'openai-agents[voice]' sounddevice numpy pydantic")
    exit(1)

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable or add it to your .env file")
    exit(1)

# =============================================================================
# TOOLS FOR PRESCHOOL READING
# =============================================================================

class StudentProfile(BaseModel):
    """Model for student information"""
    name: str
    age: int
    interests: list
    learning_style: str
    current_lesson: str

@function_tool
def get_student_profile(name: str) -> Dict[str, Any]:
    """Get student profile with interests and learning preferences"""
    profiles = {
        "Emma": {
            "age": 4, 
            "interests": ["unicorns", "rainbows", "dancing", "colors"],
            "learning_style": "visual",
            "current_lesson": "letter sounds",
            "likes": ["sparkly things", "music", "animals"],
            "dislikes": ["loud noises", "scary stories"]
        },
        "Liam": {
            "age": 5,
            "interests": ["dinosaurs", "trucks", "superheroes", "building"],
            "learning_style": "kinesthetic", 
            "current_lesson": "sight words",
            "likes": ["action", "adventure", "being strong"],
            "dislikes": ["sitting still too long", "quiet activities"]
        },
        "Sophia": {
            "age": 6,
            "interests": ["fairy tales", "friendship", "art", "nature"],
            "learning_style": "auditory",
            "current_lesson": "reading comprehension", 
            "likes": ["stories", "helping others", "creative activities"],
            "dislikes": ["rushing", "competition"]
        }
    }
    
    return profiles.get(name, {
        "age": 4,
        "interests": ["learning", "stories", "fun"],
        "learning_style": "visual",
        "current_lesson": "beginning reading",
        "likes": ["encouragement", "games"],
        "dislikes": ["feeling confused"]
    })

@function_tool
def create_personalized_story(lesson_topic: str, student_name: str) -> Dict[str, Any]:
    """Create a story that incorporates student interests and lesson objectives"""
    profile = get_student_profile(student_name)
    interests = profile.get("interests", ["adventure"])
    
    story_templates = {
        "letter_b": {
            "unicorns": "Once upon a time, a beautiful unicorn named Bella found a magical BOOK. The BALL she bounced sparkled like a rainbow as she said 'B-B-B!'",
            "dinosaurs": "Brave Brontosaurus loved to BOUNCE his big BALL. 'B-B-B!' he roared as he built a BRIDGE with BLOCKS.",
            "fairy_tales": "Beautiful Belle the fairy found a BUTTERFLY on a BANANA tree. She opened her favorite BOOK and read about BEARS."
        },
        "sight_words": {
            "unicorns": "THE magical unicorn AND her friend went TO a sparkling castle. SAID the unicorn, 'YOU are MY best friend!'",
            "dinosaurs": "THE mighty T-Rex AND his friends went TO a volcano. 'YOU are strong!' SAID the wise dinosaur.",
            "fairy_tales": "THE princess AND her friend went TO the enchanted forest. 'WE will help!' SAID the kind fairy."
        }
    }
    
    # Find matching story
    main_interest = interests[0] if interests else "adventure"
    if lesson_topic in story_templates and main_interest in story_templates[lesson_topic]:
        story = story_templates[lesson_topic][main_interest]
    else:
        story = f"Once upon a time, a brave student loved to learn about {lesson_topic}. They practiced every day and became very smart!"
    
    return {
        "story": story,
        "lesson_focus": lesson_topic,
        "personalization": f"Story created for {student_name} based on their love of {main_interest}",
        "practice_words": ["THE", "AND", "TO", "SAID", "YOU", "MY"] if lesson_topic == "sight_words" else ["B", "BALL", "BOOK", "BEAUTIFUL"]
    }

@function_tool
def create_pronunciation_guide(sound: str, difficulty_reason: str) -> Dict[str, Any]:
    """Create detailed pronunciation help for specific sounds"""
    guides = {
        "b": {
            "mouth_position": "Press your lips together, then let them pop open",
            "demonstration": "Watch my lips: B-B-B. See how they come together and pop apart?",
            "practice_steps": ["Put your lips together", "Build up air behind them", "Let them pop open with a 'B' sound"],
            "encouragement": "You're doing great! This sound can be tricky at first."
        },
        "th": {
            "mouth_position": "Put your tongue between your teeth, then blow air gently",
            "demonstration": "Look at my tongue - it peeks out just a little bit",
            "practice_steps": ["Stick your tongue out just a tiny bit", "Put it between your teeth", "Blow air gently"],
            "encouragement": "This is one of the hardest sounds - you're brave for trying!"
        },
        "r": {
            "mouth_position": "Curl your tongue back without touching the roof of your mouth",
            "demonstration": "My tongue is like a little wave, curved but not touching",
            "practice_steps": ["Make your tongue into a curve", "Don't let it touch the top", "Make a growling sound"],
            "encouragement": "The R sound takes lots of practice - keep going!"
        }
    }
    
    return guides.get(sound, {
        "mouth_position": "Let's practice this sound step by step",
        "demonstration": "Watch how I make this sound",
        "practice_steps": ["Listen to the sound", "Try to copy it", "Practice slowly"],
        "encouragement": "Every sound is learnable with practice!"
    })

@function_tool
def create_learning_quiz(topic: str, difficulty: str) -> Dict[str, Any]:
    """Create an engaging quiz or game for assessment"""
    quizzes = {
        "letter_sounds": {
            "easy": {
                "format": "Sound Hunt Game",
                "questions": [
                    "What sound does B make?",
                    "Can you find something that starts with M?", 
                    "What letter makes the 'sss' sound?"
                ],
                "game_element": "We're going on a letter sound treasure hunt!"
            },
            "medium": {
                "format": "Rhyming Game", 
                "questions": [
                    "What rhymes with 'cat'?",
                    "Can you make three words that start with 'B'?",
                    "What sound is the same in 'dog' and 'dig'?"
                ],
                "game_element": "Let's play the rhyming magic game!"
            }
        },
        "sight_words": {
            "easy": {
                "format": "Word Detective",
                "questions": [
                    "Can you find the word 'THE' in this sentence?",
                    "Point to the word 'AND'",
                    "Which word says 'YOU'?"
                ],
                "game_element": "You're a word detective solving mysteries!"
            }
        }
    }
    
    return quizzes.get(topic, {}).get(difficulty, {
        "format": "Learning Check",
        "questions": ["What did we learn today?", "Can you try that again?", "What was your favorite part?"],
        "game_element": "Let's see how much you learned!"
    })

@function_tool
def simplify_concept(original_concept: str, confusion_area: str, student_name: str) -> Dict[str, Any]:
    """Simplify and reframe concepts when student is confused"""
    profile = get_student_profile(student_name)
    interests = profile.get("interests", ["fun"])
    learning_style = profile.get("learning_style", "visual")
    
    simplifications = {
        "letter_sounds": {
            "visual": "Think of the letter like a picture - B looks like a bouncing ball!",
            "kinesthetic": "Let's move our body like the letter - make your arms round like a B!",
            "auditory": "Listen to the sound the letter makes - it's like a bubble popping!"
        },
        "sight_words": {
            "visual": "This word is like a special picture you remember with your eyes",
            "kinesthetic": "Let's spell it in the air with our finger - nice and big!",
            "auditory": "This word has a special rhythm - let's clap it out!"
        },
        "blending": {
            "visual": "Sounds are like puzzle pieces that fit together",
            "kinesthetic": "Let's slide the sounds together like toy cars connecting",
            "auditory": "Sounds want to hold hands and make a word together"
        }
    }
    
    style_approach = simplifications.get(original_concept, {}).get(learning_style, "Let's try a different way to think about this")
    
    # Add interest-based metaphor
    interest_metaphor = ""
    if "unicorns" in interests:
        interest_metaphor = "like magical unicorn spells"
    elif "dinosaurs" in interests:
        interest_metaphor = "like how dinosaurs communicate"
    elif "fairy_tales" in interests:
        interest_metaphor = "like fairy tale magic"
    
    return {
        "simplified_explanation": style_approach,
        "interest_connection": interest_metaphor,
        "new_approach": f"Let's think of it {interest_metaphor} - {style_approach}",
        "encouragement": "Sometimes we need to look at things in a new way, and that's perfectly okay!"
    }

# =============================================================================
# SPECIALIZED AGENTS FOR ADVANCED PRESCHOOL READING
# =============================================================================

# Base Patient Teacher System Prompt - Based on OpenAI's guidance
base_teacher_prompt = """
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

# 1. ENCOURAGER AGENT
encourager_agent = Agent(
    name="Encourager",
    handoff_description="Specialist for motivation, praise, and building confidence when students feel discouraged",
    instructions=base_teacher_prompt + """
    
    You are the encouragement specialist! Your job is to:
    - Boost confidence when students feel frustrated or sad
    - Celebrate every small achievement with genuine enthusiasm
    - Help students feel proud of their efforts, not just results
    - Remind them that learning takes practice and mistakes are okay
    - Use lots of positive energy and cheerful language
    
    Always focus on effort over perfection. Make every student feel special and capable!
    """,
    tools=[]
)

# 2. PRONUNCIATION HELPER AGENT
pronunciation_helper_agent = Agent(
    name="PronunciationHelper", 
    handoff_description="Specialist for helping when students mispronounce sounds or need pronunciation guidance",
    instructions=base_teacher_prompt + """
    
    You are the pronunciation specialist! When students struggle with sounds:
    - Always stay polite and gentle - never make them feel bad
    - Use the create_pronunciation_guide tool to provide step-by-step help
    - Break down the mouth movements clearly and simply
    - Give them encouragement that the sound is learnable
    - Ask them to try again with a positive, patient tone
    - Celebrate when they improve, even if not perfect
    
    Remember: pronunciation is hard work, and every attempt deserves praise!
    """,
    tools=[create_pronunciation_guide]
)

# 3. STORY TELLER TEACHER AGENT (Main Teaching)
story_teller_agent = Agent(
    name="StoryTellerTeacher",
    handoff_description="Main teacher who creates personalized stories based on student interests to teach lessons",
    instructions=base_teacher_prompt + """
    
    You are the main story-telling teacher! You make learning magical by:
    - Using the create_personalized_story tool to craft stories based on each student's interests
    - Using the get_student_profile tool to understand what each child loves
    - Weaving lesson objectives into exciting, personalized narratives
    - Making every lesson feel like an adventure tailored just for them
    - Connecting reading skills to things they care about (dinosaurs, unicorns, etc.)
    
    Remember: When learning feels personal and fun, children remember it better!
    """,
    tools=[create_personalized_story, get_student_profile]
)

# 4. TESTER AGENT
tester_agent = Agent(
    name="Tester",
    handoff_description="Specialist for assessing student knowledge through fun games, quizzes, and interactive tests",
    instructions=base_teacher_prompt + """
    
    You are the testing specialist! You assess learning through fun:
    - Use the create_learning_quiz tool to make engaging assessments
    - Turn tests into games and adventures, never boring drills
    - Focus on what they KNOW, not what they missed
    - Create challenges that feel like play, not pressure
    - Give immediate positive feedback for effort
    - Help them see assessment as a chance to show off their learning
    
    Make testing feel like the most fun part of learning!
    """,
    tools=[create_learning_quiz]
)

# 5. SIMPLIFIER AGENT
simplifier_agent = Agent(
    name="Simplifier",
    handoff_description="Specialist for breaking down confusing concepts and finding new ways to explain when students don't understand",
    instructions=base_teacher_prompt + """
    
    You are the simplification specialist! When students are confused:
    - Use the simplify_concept tool to reframe ideas in new ways
    - Use the get_student_profile tool to understand their learning style and interests
    - Find creative analogies and metaphors that relate to what they love
    - Break complex ideas into tiny, manageable steps
    - Offer multiple ways to think about the same concept
    - Never make them feel bad for not understanding the first time
    
    Every child can learn - sometimes we just need to find the right way to explain it!
    """,
    tools=[simplify_concept, get_student_profile]
)

# 6. MAIN TRIAGE AGENT - Routes to appropriate specialist
main_teacher_agent = Agent(
    name="MainTeacher",
    instructions=prompt_with_handoff_instructions(f"""
    {base_teacher_prompt}
    
    You are the main preschool reading teacher! You warmly greet students and route them to the right specialist.
    
    Based on what the student needs, route them to:
    - Encourager: when they seem discouraged, frustrated, or need motivation
    - PronunciationHelper: when they mispronounce sounds or need help with speech
    - StoryTellerTeacher: for main lessons using personalized stories (most common)
    - Tester: when they're ready to show what they've learned through games/quizzes
    - Simplifier: when they're confused and need concepts explained differently
    
    Always start with a warm greeting and figure out what kind of help they need today!
    """),
    handoffs=[encourager_agent, pronunciation_helper_agent, story_teller_agent, tester_agent, simplifier_agent]
)

# =============================================================================
# VOICE PIPELINE CONFIGURATION
# =============================================================================

async def run_text_example():
    """Run a text-based example to show the new chained agents in action"""
    print("=== PRESCHOOL READING AI - TEXT EXAMPLE ===\n")
    print("🎯 Demonstrating the new specialized agent system:")
    print("   • Encourager - motivation and confidence building") 
    print("   • PronunciationHelper - speech assistance")
    print("   • StoryTellerTeacher - personalized story-based lessons")
    print("   • Tester - fun assessments and games")
    print("   • Simplifier - concept clarification\n")
    
    test_queries = [
        "I'm feeling sad because reading is too hard for me",  # → Encourager
        "I can't say the 'th' sound correctly",  # → PronunciationHelper  
        "I want to learn about dinosaurs and letters for Liam",  # → StoryTellerTeacher
        "Can you test me on what I learned about the letter B?",  # → Tester
        "I don't understand what a sight word is",  # → Simplifier
    ]
    
    with trace("Advanced Preschool Reading Demo"):
        for i, query in enumerate(test_queries, 1):
            print(f"📝 Example {i}:")
            print(f"👶 Child: {query}")
            result = await Runner.run(main_teacher_agent, query)
            print(f"👩‍🏫 Teacher: {result.final_output}")
            print("-" * 60)
            print()

async def run_voice_example():
    """Run the voice-based chained agent system"""
    print("\n=== 🎤 PRESCHOOL READING AI - VOICE MODE ===")
    print("🎧 IMPORTANT: Make sure you have:")
    print("   • Working microphone (for your voice)")
    print("   • Speakers or headphones (to hear the teacher)")
    print("   • Quiet environment")
    print("")
    print("📋 HOW IT WORKS:")
    print("   1. Press Enter to start recording your voice")
    print("   2. Speak your question (e.g., 'Help me with the letter B')")
    print("   3. Press Enter again to stop recording")
    print("   4. The AI teacher will respond with voice!")
    print("   5. Type 'quit' to exit")
    print("")
    
    # Test audio devices
    try:
        samplerate = int(sd.query_devices(kind='input')['default_samplerate'])
        print(f"✅ Audio ready (sample rate: {samplerate} Hz)")
    except Exception as e:
        print(f"⚠️ Audio setup issue: {e}")
        print("   Continuing anyway...")
        samplerate = 44100
    
    print("\n" + "="*50)
        
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
    
    # Ask user what they want to do
    print("\nChoose your experience:")
    print("1. 🎤 Voice Mode (speak and hear the AI teacher)")
    print("2. 💬 Text Demo (see how agents work)")
    print("3. 🎯 Both (text demo first, then voice)")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        await run_voice_example()
    elif choice == "2":
        await run_text_example()
    elif choice == "3":
        await run_text_example()
        voice_choice = input("\nReady for voice mode? (y/n): ").lower()
        if voice_choice == 'y':
            await run_voice_example()
    else:
        print("Invalid choice. Starting voice mode...")
        await run_voice_example()
    
    print("\n✨ Thank you for trying the Advanced Preschool Reading AI!")
    print("This demo shows how specialized chained agents work together:")
    print("• 🎯 Intelligent routing to the right specialist for each need")
    print("• 🤗 Encourager for motivation and confidence building") 
    print("• 🗣️ PronunciationHelper for speech correction and guidance")
    print("• 📚 StoryTellerTeacher for personalized, interest-based lessons")
    print("• 🎮 Tester for fun assessments and educational games")
    print("• 💡 Simplifier for breaking down confusing concepts")
    print("\nThis creates a truly adaptive and personalized learning experience!")

if __name__ == "__main__":
    asyncio.run(main()) 