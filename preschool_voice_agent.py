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
    from database import db  # Import our learning database
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
    """Get comprehensive student profile from database with learning analytics"""
    return db.get_student_profile(name)

@function_tool 
def get_current_lesson_plan(student_name: str) -> Dict[str, Any]:
    """Get the current active lesson plan for a student"""
    plan = db.get_current_lesson_plan(student_name)
    if plan:
        return plan
    else:
        return {
            "message": f"No active lesson plan found for {student_name}",
            "learning_objective": "Alphabet recognition and letter sounds",
            "lesson_steps": ["Assess current knowledge", "Introduce new letters", "Practice sounds", "Review and reinforce"],
            "target_skills": ["letter recognition", "phonemic awareness"],
            "personalization_notes": "Create lesson plan using lesson planner"
        }

@function_tool
def record_learning_session(student_name: str, lesson_topic: str, agent_used: str, 
                          effectiveness_rating: int, session_notes: str) -> Dict[str, Any]:
    """Record details about a learning session for analysis"""
    db.add_learning_session(
        student_name=student_name,
        lesson_topic=lesson_topic, 
        agent_used=agent_used,
        conversation_summary=session_notes,
        effectiveness=effectiveness_rating,
        notes=""
    )
    return {"status": "Session recorded successfully", "student": student_name}

@function_tool
def add_student_accomplishment(student_name: str, achievement: str, skill_category: str, 
                             confidence_level: int) -> Dict[str, Any]:
    """Add a learning accomplishment to the student's record"""
    db.add_accomplishment(student_name, achievement, skill_category, confidence_level)
    return {"status": "Accomplishment recorded", "achievement": achievement}

@function_tool
def update_student_learning_data(student_name: str, learning_observations: Dict[str, Any]) -> Dict[str, Any]:
    """Update student profile with new learning insights"""
    db.update_student_profile(student_name, learning_observations)
    return {"status": "Profile updated", "student": student_name}

@function_tool
def create_new_lesson_plan(student_name: str, learning_objective: str, lesson_steps: list, 
                          target_skills: list, personalization_notes: str) -> Dict[str, Any]:
    """Create a personalized lesson plan for a student"""
    plan_id = db.create_lesson_plan(
        student_name=student_name,
        learning_objective=learning_objective, 
        lesson_steps=lesson_steps,
        target_skills=target_skills,
        personalization_notes=personalization_notes
    )
    return {
        "status": "Lesson plan created",
        "plan_id": plan_id,
        "objective": learning_objective,
        "student": student_name
    }

@function_tool
def get_parent_dashboard_data(student_name: str) -> Dict[str, Any]:
    """Get comprehensive dashboard data for parents"""
    return db.get_parent_dashboard(student_name)

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

# 6. LESSON EVALUATOR AGENT (Text-only, Analytics)
lesson_evaluator_agent = Agent(
    name="LessonEvaluator",
    instructions="""
    You are the lesson evaluation specialist! Your job is to analyze learning conversations and extract insights.
    
    RESPONSIBILITIES:
    - Monitor conversations between students and teaching agents
    - Assess learning effectiveness (1-5 scale) based on student responses
    - Identify what teaching methods work best for each student
    - Update student profiles with learning insights
    - Record accomplishments and progress milestones
    - Provide mid-lesson feedback to other agents about what's working
    
    ANALYSIS FOCUS:
    - Does the student seem engaged and understanding?
    - What teaching style is most effective (visual, auditory, kinesthetic)?
    - What interests/topics motivate this student most?
    - Are there any confusion patterns or challenging areas?
    - What accomplishments should be celebrated?
    
    Use tools to record sessions, update profiles, and track accomplishments.
    """,
    tools=[record_learning_session, update_student_learning_data, add_student_accomplishment, get_student_profile]
)

# 7. LESSON PLANNER AGENT (Text-only, Planning)
lesson_planner_agent = Agent(
    name="LessonPlanner", 
    instructions="""
    You are the lesson planning specialist! Your job is to create personalized learning paths.
    
    PRIMARY LEARNING OBJECTIVE: Alphabet recognition and letter sounds (phonemic awareness)
    
    RESPONSIBILITIES:
    - Analyze student profiles to understand learning preferences and progress
    - Create step-by-step lesson plans toward alphabet mastery
    - Personalize lessons based on interests (dinosaurs, unicorns, fairy tales, etc.)
    - Plan progression from current level to alphabet fluency
    - Consider learning style (visual, auditory, kinesthetic) in lesson design
    - Set realistic, achievable milestones
    
    LESSON PLAN STRUCTURE:
    1. Current assessment (where is the student now?)
    2. Specific learning objective (what will they learn today?)
    3. Personalized approach (how does it connect to their interests?)
    4. Step-by-step activities (concrete actions)
    5. Practice reinforcement (how to solidify learning)
    6. Assessment method (how to check understanding)
    
    Use tools to get student data and create detailed lesson plans.
    """,
    tools=[get_student_profile, create_new_lesson_plan, get_parent_dashboard_data]
)

# 8. ENHANCED MAIN TRIAGE AGENT - Routes to appropriate specialist with lesson plan awareness
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
    
    Always start with a warm greeting, check their lesson plan, and figure out what help they need today!
    """),
    handoffs=[encourager_agent, pronunciation_helper_agent, story_teller_agent, tester_agent, simplifier_agent],
    tools=[get_current_lesson_plan, get_student_profile]
)

# =============================================================================
# VOICE PIPELINE CONFIGURATION
# =============================================================================

async def setup_demo_lesson_plans():
    """Create demo lesson plans for the system"""
    print("🏗️ Setting up demo lesson plans and student profiles...\n")
    
    # Create lesson plans for demo students
    demo_students = ["Emma", "Liam", "Sophia"]
    
    for student in demo_students:
        # Let the lesson planner create a plan
        planner_query = f"Create a lesson plan for {student} focusing on alphabet recognition and letter sounds"
        await Runner.run(lesson_planner_agent, planner_query)
    
    print("✅ Demo lesson plans created!\n")

async def run_text_example():
    """Run a text-based example to show the new advanced learning system"""
    print("=== ADVANCED PRESCHOOL READING AI - FULL SYSTEM DEMO ===\n")
    print("🎯 Demonstrating the complete adaptive learning system:")
    print("   • 🤗 Encourager - motivation and confidence building") 
    print("   • 🗣️ PronunciationHelper - speech assistance")
    print("   • 📚 StoryTellerTeacher - personalized story-based lessons")
    print("   • 🎮 Tester - fun assessments and games")
    print("   • 💡 Simplifier - concept clarification")
    print("   • 📊 LessonEvaluator - learning analytics (background)")
    print("   • 📋 LessonPlanner - curriculum planning (background)")
    print("   • 💾 Database - persistent learning profiles\n")
    
    # Setup demo data
    await setup_demo_lesson_plans()
    
    test_queries = [
        ("Emma", "I'm feeling sad because reading is too hard for me"),  # → Encourager
        ("Liam", "I can't say the 'th' sound correctly"),  # → PronunciationHelper  
        ("Emma", "I want to learn about unicorns and the letter B"),  # → StoryTellerTeacher
        ("Sophia", "Can you test me on what I learned about letters?"),  # → Tester
        ("Liam", "I don't understand what a sight word is"),  # → Simplifier
    ]
    
    with trace("Advanced Learning System Demo"):
        for i, (student_name, query) in enumerate(test_queries, 1):
            print(f"📝 Example {i} - Student: {student_name}")
            print(f"👶 Child: {query}")
            
            # Main teaching interaction
            result = await Runner.run(main_teacher_agent, f"Student {student_name} says: {query}")
            print(f"👩‍🏫 Teacher: {result.final_output}")
            
            # Lesson evaluator analyzes the session (background)
            evaluation_query = f"Analyze this learning session: Student {student_name} asked '{query}' and received teaching. Rate effectiveness and update profile."
            eval_result = await Runner.run(lesson_evaluator_agent, evaluation_query)
            print(f"📊 Analysis: {eval_result.final_output}")
            
            print("-" * 70)
            print()

async def demo_parent_dashboard():
    """Show parent dashboard functionality"""
    print("=== 👨‍👩‍👧‍👦 PARENT DASHBOARD DEMO ===\n")
    
    students = ["Emma", "Liam", "Sophia"]
    
    for student in students:
        print(f"📊 Dashboard for {student}:")
        dashboard_data = db.get_parent_dashboard(student)
        
        print(f"   Age: {dashboard_data['profile'].get('age', 'Unknown')}")
        print(f"   Learning Style: {dashboard_data['profile'].get('learning_style', 'Unknown')}")
        print(f"   Interests: {', '.join(dashboard_data['profile'].get('interests', []))}")
        
        if dashboard_data['recent_sessions']:
            print(f"   Recent Sessions: {len(dashboard_data['recent_sessions'])}")
            latest = dashboard_data['recent_sessions'][0]
            print(f"   Latest Topic: {latest['topic']} (Effectiveness: {latest['effectiveness']}/5)")
        
        if dashboard_data['skill_progress']:
            print("   Skill Progress:")
            for skill in dashboard_data['skill_progress']:
                print(f"     • {skill['category']}: {skill['achievements_count']} achievements (Confidence: {skill['average_confidence']}/5)")
        
        print("-" * 50)
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
    """Main function to demonstrate the advanced learning system"""
    print("🎓 ADVANCED PRESCHOOL READING AI - INTELLIGENT LEARNING SYSTEM")
    print("Based on OpenAI's Agents SDK with Advanced Learning Analytics")
    print("=" * 70)
    
    # Ask user what they want to do
    print("\nChoose your experience:")
    print("1. 🎤 Voice Mode (speak and hear the AI teacher)")
    print("2. 💬 Full System Demo (see all agents working together)")
    print("3. 👨‍👩‍👧‍👦 Parent Dashboard (see learning analytics)")
    print("4. 🎯 Complete Experience (demo + dashboard + voice)")
    
    choice = input("\nEnter your choice (1, 2, 3, or 4): ").strip()
    
    if choice == "1":
        await run_voice_example()
    elif choice == "2":
        await run_text_example()
    elif choice == "3":
        await demo_parent_dashboard()
    elif choice == "4":
        await run_text_example()
        await demo_parent_dashboard()
        voice_choice = input("\nReady for voice mode? (y/n): ").lower()
        if voice_choice == 'y':
            await run_voice_example()
    else:
        print("Invalid choice. Starting full system demo...")
        await run_text_example()
    
    print("\n✨ Thank you for trying the Advanced Preschool Reading AI!")
    print("This intelligent learning system features:")
    print("• 🎯 Smart routing to specialized teaching agents")
    print("• 🤗 Encourager for motivation and confidence building") 
    print("• 🗣️ PronunciationHelper for speech correction and guidance")
    print("• 📚 StoryTellerTeacher for personalized, interest-based lessons")
    print("• 🎮 Tester for fun assessments and educational games")
    print("• 💡 Simplifier for breaking down confusing concepts")
    print("• 📊 LessonEvaluator for continuous learning analytics")
    print("• 📋 LessonPlanner for adaptive curriculum planning")
    print("• 💾 Persistent database tracking individual progress")
    print("• 👨‍👩‍👧‍👦 Parent dashboard with detailed learning insights")
    print("\nThis creates a truly intelligent, adaptive, and personalized learning experience!")

if __name__ == "__main__":
    asyncio.run(main()) 