# Preschool Reading AI - Chained Voice Agent Example

This is a complete working example of OpenAI's **Chained Voice Agent** architecture, specifically designed for preschool reading education. It demonstrates the key concepts from OpenAI's documentation including multi-agent orchestration, handoffs, and voice processing with **Patient Teacher** instructions.

## 🎯 What This Example Demonstrates

### Chained Agent Architecture
- **Main Teacher Agent**: Acts as a triage agent that routes requests to specialists
- **Phonics Specialist**: Handles letter sounds, phonics, and pronunciation practice
- **Sight Words Specialist**: Manages high-frequency word recognition and practice
- **Progress Tracker**: Tracks reading progress and celebrates achievements

### Patient Teacher Instructions
The agents are designed with specialized prompts that create:
- Warm, encouraging personality appropriate for ages 3-6
- Patient, nurturing teaching approach
- Voice-optimized responses (slow, clear speech with enthusiasm)
- Educational best practices for early readers

### Voice Processing Pipeline
- **Speech-to-Text**: Converts child's voice to text
- **Agent Processing**: Routes through appropriate specialist agents
- **Text-to-Speech**: Converts response back to natural speech

## 🛠 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Your OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

### 3. Run the Example
```bash
python preschool_voice_agent.py
```

## 🎮 How to Use

### Text Mode (Default)
The example will first run in text mode to demonstrate the chained agents:
- Shows how different queries are routed to appropriate specialists
- Displays agent responses in console
- Demonstrates the handoff mechanism between agents

### Voice Mode (Optional)
After the text demo, you can try voice mode:
1. Press Enter to start recording
2. Speak your question (e.g., "I want to practice the letter B sound")
3. Press Enter again to stop recording
4. The AI teacher will respond with voice
5. Type 'quit' to exit

## 🔧 Architecture Overview

```
Child's Voice Input
        ↓
   Speech-to-Text
        ↓
   Main Teacher Agent (Triage)
        ↓
   ┌─────────────────────────────────┐
   │  Routes to Appropriate Agent:   │
   │  • PhonicsTeacher              │
   │  • SightWordsTeacher           │
   │  • ProgressTracker             │
   └─────────────────────────────────┘
        ↓
   Specialist Agent Processing
        ↓
   Text-to-Speech
        ↓
   Audio Response to Child
```

## 📚 Key Features

### Custom Tools
- **get_reading_progress()**: Tracks individual child progress
- **get_sight_words()**: Provides difficulty-appropriate word lists
- **create_phonics_exercise()**: Generates phonics practice activities

### Patient Teacher Prompting
- Warm, encouraging personality
- Age-appropriate language (3-6 years)
- Voice-optimized responses
- Educational best practices

### Handoff System
- Intelligent routing based on child's needs
- Seamless transitions between specialists
- Maintains context across handoffs

## 🎯 Example Interactions

**Child**: "I want to practice the letter B sound"  
**System**: Routes to PhonicsTeacher → Provides B sound exercises

**Child**: "Can you check Emma's reading progress?"  
**System**: Routes to ProgressTracker → Shows progress and celebrates achievements

**Child**: "I need help with sight words for beginners"  
**System**: Routes to SightWordsTeacher → Provides beginner word list

## 📊 Tracing and Monitoring

The example includes built-in tracing that allows you to:
- View agent handoff decisions
- Monitor tool usage
- Track conversation flow
- Debug and optimize performance

View traces at: https://platform.openai.com/traces

## 🔄 Customization Options

### Adding New Specialists
```python
new_specialist = Agent(
    name="NewSpecialist",
    handoff_description="Specialist for new functionality",
    instructions=patient_teacher_prompt + "Your specific instructions...",
    tools=[your_custom_tools]
)

# Add to main teacher's handoffs
main_teacher_agent.handoffs.append(new_specialist)
```

### Creating Custom Tools
```python
@function_tool
def your_custom_tool(parameter: str) -> Dict[str, Any]:
    """Your tool description"""
    # Your implementation
    return {"result": "value"}
```

### Modifying Voice Settings
The voice pipeline can be customized for different:
- Speaking pace
- Voice personality
- Audio quality
- Language settings

## 🎓 Educational Applications

This example can be extended for:
- **Phonics Programs**: Systematic phonics instruction
- **Sight Word Practice**: High-frequency word recognition
- **Reading Assessment**: Progress tracking and evaluation
- **Individualized Learning**: Adaptive difficulty levels
- **Parent Communication**: Progress reports and home activities

## 🔍 Technical Details

### Based on OpenAI Documentation
This example implements concepts from:
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-js/)
- [Voice Agents Documentation](https://openai.github.io/openai-agents-js/guides/voice-agents/)
- [Building Voice Assistants with Agents SDK](https://cookbook.openai.com/examples/agents_sdk/app_assistant_voice_agents)

### Architecture Patterns
- **Chain of Responsibility**: For agent routing
- **Strategy Pattern**: For different teaching approaches
- **Observer Pattern**: For progress tracking
- **Factory Pattern**: For tool creation

## 🚀 Next Steps

To extend this example:
1. Add more specialized agents (story reading, writing practice, etc.)
2. Implement persistent progress tracking with databases
3. Add parent/teacher dashboards
4. Create multimedia content integration
5. Implement assessment and reporting features

## 📝 License

This example is provided for educational purposes and demonstration of OpenAI's Agents SDK capabilities. 