#!/usr/bin/env python3
"""
Configuration file for Preschool Reading AI - Chained Voice Agent
Customize these settings for your specific needs
"""

# =============================================================================
# VOICE SETTINGS
# =============================================================================

# Audio settings
SAMPLE_RATE = 44100  # Audio sample rate (Hz)
CHANNELS = 1  # Mono audio
AUDIO_FORMAT = 'int16'  # Audio format

# Voice characteristics for Text-to-Speech
VOICE_SETTINGS = {
    "voice": "alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
    "speed": 0.9,      # Speaking speed (0.25 to 4.0)
    "pitch": 1.0,      # Voice pitch (0.5 to 2.0)
}

# =============================================================================
# AGENT SETTINGS
# =============================================================================

# Model selection
DEFAULT_MODEL = "gpt-4o-mini"  # Options: gpt-4o-mini, gpt-4o, gpt-4-turbo

# Agent response settings
MAX_RESPONSE_LENGTH = 200  # Maximum characters in response
RESPONSE_TIMEOUT = 30  # Seconds to wait for response

# =============================================================================
# EDUCATIONAL CONTENT
# =============================================================================

# Reading levels and progression
READING_LEVELS = {
    "Pre-Reader": {
        "age_range": "3-4 years",
        "skills": ["letter recognition", "phonemic awareness", "print concepts"],
        "sight_words": ["I", "me", "my", "you", "the", "a"]
    },
    "Beginning Reader": {
        "age_range": "4-5 years", 
        "skills": ["letter sounds", "simple words", "basic phonics"],
        "sight_words": ["and", "to", "said", "you", "of", "we", "my", "be", "have", "from"]
    },
    "Early Reader": {
        "age_range": "5-6 years",
        "skills": ["word families", "simple sentences", "reading fluency"],
        "sight_words": ["they", "know", "want", "been", "good", "much", "some", "time", "very", "when"]
    },
    "Developing Reader": {
        "age_range": "6-7 years",
        "skills": ["complex words", "reading comprehension", "story structure"],
        "sight_words": ["would", "there", "each", "which", "their", "called", "first", "water", "after", "back"]
    }
}

# Phonics progression
PHONICS_SEQUENCE = [
    # Single consonants
    "m", "s", "t", "a", "n", "p", "i", "c", "k", "e", "r", "d",
    # More single sounds
    "h", "u", "l", "f", "b", "j", "o", "g", "w", "v", "x", "y", "z", "q",
    # Consonant blends
    "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sl", "sm", "sn", "sp", "st", "sw", "tr",
    # Vowel sounds
    "ai", "ay", "ea", "ee", "ey", "ie", "oa", "oe", "oo", "ou", "ow", "ue", "ui"
]

# =============================================================================
# PERSONALIZATION SETTINGS
# =============================================================================

# Child profiles (customize with actual student data)
CHILD_PROFILES = {
    "Emma": {
        "age": 4,
        "level": "Beginning Reader",
        "interests": ["animals", "stories", "colors"],
        "learning_style": "visual",
        "progress": {
            "words_learned": 45,
            "books_completed": 8,
            "current_phonics": "b",
            "last_session": "2024-01-15"
        }
    },
    "Liam": {
        "age": 5,
        "level": "Early Reader", 
        "interests": ["dinosaurs", "cars", "adventure"],
        "learning_style": "kinesthetic",
        "progress": {
            "words_learned": 78,
            "books_completed": 15,
            "current_phonics": "tr",
            "last_session": "2024-01-16"
        }
    },
    "Sophia": {
        "age": 6,
        "level": "Developing Reader",
        "interests": ["fairy tales", "friendship", "art"],
        "learning_style": "auditory",
        "progress": {
            "words_learned": 120,
            "books_completed": 22,
            "current_phonics": "oa",
            "last_session": "2024-01-17"
        }
    }
}

# =============================================================================
# TEACHING PREFERENCES
# =============================================================================

# Encouragement phrases
ENCOURAGEMENT_PHRASES = [
    "Great job!", "You're doing wonderful!", "Keep it up!", "Excellent work!",
    "You're such a good reader!", "I'm so proud of you!", "Amazing progress!",
    "You're getting better every day!", "Fantastic effort!", "Well done!"
]

# Gentle correction phrases
CORRECTION_PHRASES = [
    "Let's try that again together.", "Almost there! Let's practice once more.",
    "That's okay, let's sound it out.", "You're learning! Let's try again.",
    "Good try! Let's break it down.", "Nearly got it! One more time."
]

# =============================================================================
# SYSTEM PREFERENCES
# =============================================================================

# Logging and monitoring
ENABLE_TRACING = True  # Enable OpenAI tracing
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
SAVE_SESSIONS = True  # Save session data for progress tracking

# Safety settings
CONTENT_FILTER = True  # Enable content filtering
SESSION_TIMEOUT = 300  # Session timeout in seconds (5 minutes)
MAX_CONSECUTIVE_ERRORS = 3  # Maximum errors before suggesting break

# =============================================================================
# CUSTOMIZATION FUNCTIONS
# =============================================================================

def get_child_profile(name: str) -> dict:
    """Get a child's profile or create default"""
    return CHILD_PROFILES.get(name, {
        "age": 4,
        "level": "Beginning Reader",
        "interests": ["learning", "stories"],
        "learning_style": "visual",
        "progress": {
            "words_learned": 0,
            "books_completed": 0,
            "current_phonics": "m",
            "last_session": None
        }
    })

def get_appropriate_sight_words(level: str) -> list:
    """Get sight words appropriate for reading level"""
    level_data = READING_LEVELS.get(level, READING_LEVELS["Beginning Reader"])
    return level_data["sight_words"]

def get_next_phonics_sound(current_sound: str) -> str:
    """Get the next phonics sound in the sequence"""
    try:
        current_index = PHONICS_SEQUENCE.index(current_sound)
        if current_index < len(PHONICS_SEQUENCE) - 1:
            return PHONICS_SEQUENCE[current_index + 1]
        else:
            return current_sound  # Already at the end
    except ValueError:
        return PHONICS_SEQUENCE[0]  # Default to first sound 