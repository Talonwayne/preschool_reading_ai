#!/usr/bin/env python3
"""
Preschool Reading AI - Simplified Voice Agent with Comprehensive Logging
Based on OpenAI's Agents SDK for voice teaching
"""

import asyncio
import os
import logging
from datetime import datetime
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
    # from database import db  # Commented out for simplified version
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install with: pip install openai-agents 'openai-agents[voice]' sounddevice numpy pydantic")
    exit(1)

# Load environment variables from .env file
load_dotenv()

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preschool_ai_session.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable or add it to your .env file")
    exit(1)

# =============================================================================
# SIMPLIFIED TOOLS FOR PRESCHOOL READING
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
    """Get student profile (simplified version)"""
    logger.info(f"🔍 Getting student profile for: {name}")
    profile = {
        "name": name,
        "age": 4,
        "interests": ["learning", "stories", "fun"],
        "learning_style": "visual",
        "current_lesson": "alphabet"
    }
    logger.info(f"📋 Student profile retrieved: {profile}")
    return profile

@function_tool 
def get_current_lesson_plan(student_name: str) -> Dict[str, Any]:
    """Get the current lesson plan (simplified version)"""
    logger.info(f"📚 Getting lesson plan for: {student_name}")
    plan = {
        "learning_objective": "Alphabet recognition and letter sounds",
        "lesson_steps": ["Learn letters A, B, C", "Practice sounds", "Have fun!"],
        "target_skills": ["letter recognition", "phonemic awareness"],
        "personalization_notes": "Focus on fun and encouragement"
    }
    logger.info(f"📖 Lesson plan retrieved: {plan}")
    return plan

@function_tool
def create_personalized_story(lesson_topic: str, student_name: str) -> Dict[str, Any]:
    """Create a story that incorporates student interests and lesson objectives"""
    logger.info(f"📖 Creating story for {student_name} about: {lesson_topic}")
    
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
    main_interest = "adventure"
    if lesson_topic in story_templates and main_interest in story_templates[lesson_topic]:
        story = story_templates[lesson_topic][main_interest]
    else:
        story = f"Once upon a time, a brave student loved to learn about {lesson_topic}. They practiced every day and became very smart!"
    
    result = {
        "story": story,
        "lesson_focus": lesson_topic,
        "personalization": f"Story created for {student_name}",
        "practice_words": ["THE", "AND", "TO", "SAID", "YOU", "MY"] if lesson_topic == "sight_words" else ["B", "BALL", "BOOK", "BEAUTIFUL"]
    }
    
    logger.info(f"📚 Story created: {result}")
    return result

@function_tool
def create_pronunciation_guide(sound: str, difficulty_reason: str) -> Dict[str, Any]:
    """Create detailed pronunciation help for specific sounds"""
    logger.info(f"🗣️ Creating pronunciation guide for sound: {sound}")
    
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
    
    result = guides.get(sound, {
        "mouth_position": "Let's practice this sound step by step",
        "demonstration": "Watch how I make this sound",
        "practice_steps": ["Listen to the sound", "Try to copy it", "Practice slowly"],
        "encouragement": "Every sound is learnable with practice!"
    })
    
    logger.info(f"🗣️ Pronunciation guide created: {result}")
    return result

@function_tool
def create_learning_quiz(topic: str, difficulty: str) -> Dict[str, Any]:
    """Create an engaging quiz or game for assessment"""
    logger.info(f"🎮 Creating quiz for topic: {topic}, difficulty: {difficulty}")
    
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
    
    result = quizzes.get(topic, {}).get(difficulty, {
        "format": "Learning Check",
        "questions": ["What did we learn today?", "Can you try that again?", "What was your favorite part?"],
        "game_element": "Let's see how much you learned!"
    })
    
    logger.info(f"🎮 Quiz created: {result}")
    return result

@function_tool
def simplify_concept(original_concept: str, confusion_area: str, student_name: str) -> Dict[str, Any]:
    """Simplify and reframe concepts when student is confused"""
    logger.info(f"💡 Simplifying concept: {original_concept} for {student_name}")
    
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
    
    style_approach = simplifications.get(original_concept, {}).get("visual", "Let's try a different way to think about this")
    
    result = {
        "simplified_explanation": style_approach,
        "interest_connection": "like magical learning spells",
        "new_approach": f"Let's think of it like magical learning spells - {style_approach}",
        "encouragement": "Sometimes we need to look at things in a new way, and that's perfectly okay!"
    }
    
    logger.info(f"💡 Concept simplified: {result}")
    return result

# =============================================================================
# SIMPLIFIED AGENTS FOR PRESCHOOL READING
# =============================================================================

# Base Patient Teacher System Prompt - ENHANCED FOR SHORT RESPONSES
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

VOICE DELIVERY - CRITICAL FOR SHORT RESPONSES:
- Keep responses BRIEF and COMPLETE (1-2 sentences maximum)
- Speak slowly and clearly for young learners
- Use varied intonation to maintain engagement
- Pause between instructions to allow processing time
- Use excitement in your voice for achievements
- ALWAYS provide a complete thought in each response
- Avoid run-on sentences or multiple ideas in one response

RESPONSE LENGTH RULE: Every response must be 1-2 complete sentences that fully answer the student's question or need.

CONVERSATION FLOW RULE: When you ask the student a question, STOP generating immediately after the question. Wait for their response before continuing. Do not ask multiple questions in one response.
"""

# ENCOURAGER AGENT
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
    
    CRITICAL: Keep responses to 1-2 complete sentences maximum.
    CONVERSATION RULE: If you ask the student a question, stop immediately and wait for their answer.
    Always focus on effort over perfection. Make every student feel special and capable!
    """,
    tools=[]
)

# PRONUNCIATION HELPER AGENT
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
    
    CRITICAL: Keep responses to 1-2 complete sentences maximum.
    CONVERSATION RULE: If you ask the student to try a sound, stop immediately and wait for their attempt.
    Remember: pronunciation is hard work, and every attempt deserves praise!
    """,
    tools=[create_pronunciation_guide]
)

# STORY TELLER TEACHER AGENT (Main Teaching)
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
    
    CRITICAL: Keep responses to 1-2 complete sentences maximum.
    CONVERSATION RULE: If you ask the student a question about the story or lesson, stop immediately and wait for their answer.
    Remember: When learning feels personal and fun, children remember it better!
    """,
    tools=[create_personalized_story, get_student_profile]
)

# TESTER AGENT
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
    
    CRITICAL: Keep responses to 1-2 complete sentences maximum.
    CONVERSATION RULE: Ask ONE question at a time, then stop and wait for the student's answer before asking the next question.
    Make testing feel like the most fun part of learning!
    """,
    tools=[create_learning_quiz]
)

# SIMPLIFIER AGENT
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
    
    CRITICAL: Keep responses to 1-2 complete sentences maximum.
    CONVERSATION RULE: If you ask the student if they understand or want to try something, stop immediately and wait for their response.
    Every child can learn - sometimes we just need to find the right way to explain it!
    """,
    tools=[simplify_concept, get_student_profile]
)

# MAIN TEACHER AGENT - Routes to appropriate specialist
main_teacher_agent = Agent(
    name="MainTeacher",
    instructions=prompt_with_handoff_instructions(f"""
    {base_teacher_prompt}
    
    You are the main preschool reading teacher! You warmly greet students and route them to the right specialist.
    
    IMPORTANT: Use get_current_lesson_plan tool to understand today's learning objectives for this student.
    Teach toward the specific learning goals in their lesson plan!
    
    Based on what the student needs, route them to:
    - Encourager: when they seem discouraged, frustrated, or need motivation
    - PronunciationHelper: when they mispronounce sounds or need help with speech
    - StoryTellerTeacher: for main lessons using personalized stories (most common)
    - Tester: when they're ready to show what they've learned through games/quizzes
    - Simplifier: when they're confused and need concepts explained differently
    
    CRITICAL: Keep responses to 1-2 complete sentences maximum.
    CONVERSATION RULE: If you ask the student a question, stop immediately and wait for their answer.
    Always start with a warm greeting, check their lesson plan, and figure out what help they need today!
    """),
    handoffs=[encourager_agent, pronunciation_helper_agent, story_teller_agent, tester_agent, simplifier_agent],
    tools=[get_current_lesson_plan, get_student_profile]
)

# =============================================================================
# SIMPLIFIED VOICE MODE WITH COMPREHENSIVE LOGGING
# =============================================================================

async def run_simplified_voice_mode():
    """Simplified voice mode with automatic alphabet lesson on startup and comprehensive logging"""
    logger.info("🚀 Starting Preschool Reading AI - Simplified Voice Mode")
    print("\n=== 🎤 PRESCHOOL READING AI - SIMPLIFIED VOICE MODE ===")
    print("🎧 Make sure you have a microphone and speakers ready!")
    print("")
    
    # Test audio devices
    try:
        input_device = sd.query_devices(kind='input')
        samplerate = int(input_device['default_samplerate'])
        logger.info(f"✅ Audio system ready - Sample rate: {samplerate} Hz")
        print(f"✅ Audio ready (sample rate: {samplerate} Hz)")
    except (Exception, KeyError) as e:
        logger.warning(f"⚠️ Audio setup issue: {e}")
        print(f"⚠️ Audio setup issue: {e}")
        print("   Continuing anyway...")
        samplerate = 44100
    
    print("\n" + "="*50)
    
    # Start with automatic alphabet lesson
    logger.info("🎯 Starting automatic alphabet lesson")
    print("🎯 Starting with a fun alphabet lesson!")
    print("Press Enter to begin the alphabet lesson...")
    input()
    
    # Automatic alphabet lesson
    logger.info("👩‍🏫 Teacher beginning alphabet lesson")
    print("👩‍🏫 Teacher: Starting alphabet lesson...")
    
    try:
        # Simulate the teacher speaking the lesson
        lesson_text = [
            "🎵 The teacher is teaching the alphabet...",
            "📚 'Hello! Let's learn about letters today!'",
            "🔤 'A is for Apple - can you say A-A-Apple?'",
            "🅱️ 'B is for Ball - B-B-Ball!'",
            "©️ 'C is for Cat - C-C-Cat!'",
            "🎉 'Great job! You're learning your letters!'"
        ]
        
        for line in lesson_text:
            print(line)
            logger.info(f"👩‍🏫 Teacher says: {line}")
            await asyncio.sleep(0.5)
        
        # Small delay to simulate processing
        await asyncio.sleep(2)
        
    except Exception as e:
        logger.error(f"❌ Error in alphabet lesson: {e}")
        print(f"Error in alphabet lesson: {e}")
    
    print("=" * 50)
    print("🎤 Now you can ask questions! Try saying:")
    print("   • 'Help me with the letter B'")
    print("   • 'What sound does A make?'")
    print("   • 'Tell me a story about letters'")
    print("   • 'I'm feeling sad about reading'")
    print("   • 'Test me on what I learned'")
    print("=" * 50)
    
    # Simplified voice interaction loop
    session_count = 0
    while True:
        session_count += 1
        logger.info(f"🔄 Starting voice session #{session_count}")
        
        print(f"\n🎤 Press Enter to speak (or type 'quit' to exit): ", end="")
        user_input = input()
        
        if user_input.lower() == 'quit':
            logger.info("👋 User requested to quit the program")
            print("👋 Goodbye! Keep practicing your reading! 🌟")
            break
            
        logger.info(f"🎤 User input: '{user_input}'")
        print("🎤 Listening... (Press Enter when done speaking)")
        
        # Record audio
        recorded_chunks = []
        def audio_callback(indata, frames, time, status):
            recorded_chunks.append(indata.copy())
            
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', callback=audio_callback):
            input()  # Wait for Enter key
            
        if not recorded_chunks:
            logger.warning("❌ No audio recorded in this session")
            print("❌ No audio recorded. Please try again.")
            continue
            
        # Log audio recording details
        audio_duration = len(recorded_chunks) * 0.1  # Approximate duration
        logger.info(f"🎤 Audio recorded: {len(recorded_chunks)} chunks, ~{audio_duration:.1f}s duration")
        
        # Process the recording
        logger.info("🤔 Teacher is thinking about the student's question...")
        print("🤔 Teacher is thinking...")
        
        # Create voice pipeline for main interaction
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(main_teacher_agent))
        
        # Concatenate recorded audio
        recording = np.concatenate(recorded_chunks, axis=0)
        audio_input = AudioInput(buffer=recording)
        
        try:
            # Run the voice pipeline
            logger.info("🎯 Processing student's voice input through AI pipeline")
            result = await pipeline.run(audio_input)
            
            # Collect response audio and log teacher's text response
            response_chunks = []
            teacher_text_response = ""
            student_transcription = ""
            
            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    response_chunks.append(event.data)
                elif event.type == "text_stream_event":
                    # Capture the teacher's text response
                    if hasattr(event, 'data'):
                        teacher_text_response += event.data
                        logger.info(f"👩‍🏫 Teacher thinking: '{event.data}'")
                elif event.type == "transcription":
                    # Capture the student's transcription
                    if hasattr(event, 'data'):
                        student_transcription = event.data
                        logger.info(f"🎤 Student said: '{student_transcription}'")
                        print(f"🎤 Student said: '{student_transcription}'")
            
            # Log the complete teacher response
            if teacher_text_response:
                logger.info(f"👩‍🏫 Teacher wants to say: '{teacher_text_response.strip()}'")
                print(f"👩‍🏫 Teacher wants to say: '{teacher_text_response.strip()}'")
            else:
                logger.info("👩‍🏫 Teacher response text not captured")
                print("👩‍🏫 Teacher responding with voice...")
            
            if response_chunks:
                # Log the teacher's response
                logger.info(f"👩‍🏫 Teacher responding with {len(response_chunks)} audio chunks")
                print("👩‍🏫 Teacher is responding...")
                
                # Play the response
                response_audio = np.concatenate(response_chunks, axis=0)
                response_duration = len(response_audio) / samplerate
                logger.info(f"🎵 Playing teacher response: ~{response_duration:.1f}s duration")
                
                sd.play(response_audio, samplerate=samplerate)
                sd.wait()  # Wait for playback to finish
                
                logger.info("✅ Teacher response completed successfully")
            else:
                logger.warning("❌ No audio response generated by AI")
                print("❌ No audio response generated.")
                
        except Exception as e:
            logger.error(f"❌ Error processing voice: {e}")
            print(f"❌ Error processing voice: {e}")
            
        print("=" * 50)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Simplified main function with automatic alphabet lesson and comprehensive logging"""
    logger.info("🎓 Starting Preschool Reading AI - Simplified Version")
    print("🎓 PRESCHOOL READING AI - SIMPLIFIED VERSION")
    print("Based on OpenAI's Agents SDK")
    print("=" * 50)
    
    # Start with alphabet lesson automatically
    await run_simplified_voice_mode()
    
    logger.info("✨ Preschool Reading AI session completed")
    print("\n✨ Thank you for trying the Preschool Reading AI!")
    print("This system helps children learn to read through:")
    print("• 🎯 Personalized lessons based on interests")
    print("• 🤗 Encouragement and confidence building") 
    print("• 🗣️ Pronunciation help and guidance")
    print("• 📚 Fun stories and interactive learning")
    print("• 🎮 Games and assessments")
    print("• 💡 Clear explanations when confused")
    print("\nKeep practicing and have fun learning! 🌟")

if __name__ == "__main__":
    asyncio.run(main()) 