#!/usr/bin/env python3
"""
Local Database System for Preschool Reading AI
Stores student profiles, learning progress, and lesson analytics
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

class LearningDatabase:
    def __init__(self, db_path: str = "preschool_learning.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Student profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                age INTEGER,
                interests TEXT, -- JSON array
                learning_style TEXT,
                current_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Learning sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT,
                lesson_topic TEXT,
                agent_used TEXT,
                conversation_summary TEXT,
                learning_effectiveness INTEGER, -- 1-5 scale
                notes TEXT,
                session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_name) REFERENCES student_profiles (name)
            )
        ''')
        
        # Lesson plans table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lesson_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT,
                learning_objective TEXT,
                lesson_steps TEXT, -- JSON array
                target_skills TEXT, -- JSON array
                personalization_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending', -- pending, in_progress, completed
                FOREIGN KEY (student_name) REFERENCES student_profiles (name)
            )
        ''')
        
        # Learning accomplishments table (for parent dashboard)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accomplishments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT,
                achievement TEXT,
                skill_category TEXT, -- alphabet, phonics, sight_words, etc.
                date_achieved TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence_level INTEGER, -- 1-5 scale
                FOREIGN KEY (student_name) REFERENCES student_profiles (name)
            )
        ''')
        
        # Learning analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT,
                preferred_teaching_style TEXT,
                effective_strategies TEXT, -- JSON array
                challenging_areas TEXT, -- JSON array
                motivation_triggers TEXT, -- JSON array
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_name) REFERENCES student_profiles (name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_student_profile(self, name: str) -> Dict[str, Any]:
        """Get comprehensive student profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic profile
        cursor.execute('SELECT * FROM student_profiles WHERE name = ?', (name,))
        profile = cursor.fetchone()
        
        if not profile:
            # Create default profile if doesn't exist
            default_profile = {
                'name': name,
                'age': 4,
                'interests': json.dumps(['learning', 'stories']),
                'learning_style': 'visual',
                'current_level': 'beginner'
            }
            cursor.execute('''
                INSERT INTO student_profiles (name, age, interests, learning_style, current_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, 4, default_profile['interests'], 'visual', 'beginner'))
            conn.commit()
            
            profile_data = default_profile
        else:
            profile_data = {
                'name': profile[1],
                'age': profile[2],
                'interests': json.loads(profile[3]) if profile[3] else [],
                'learning_style': profile[4],
                'current_level': profile[5]
            }
        
        # Get learning analytics
        cursor.execute('SELECT * FROM learning_analytics WHERE student_name = ? ORDER BY updated_at DESC LIMIT 1', (name,))
        analytics = cursor.fetchone()
        
        if analytics:
            profile_data.update({
                'preferred_teaching_style': analytics[2],
                'effective_strategies': json.loads(analytics[3]) if analytics[3] else [],
                'challenging_areas': json.loads(analytics[4]) if analytics[4] else [],
                'motivation_triggers': json.loads(analytics[5]) if analytics[5] else []
            })
        
        # Get recent accomplishments
        cursor.execute('''
            SELECT achievement, skill_category, date_achieved, confidence_level 
            FROM accomplishments 
            WHERE student_name = ? 
            ORDER BY date_achieved DESC LIMIT 5
        ''', (name,))
        accomplishments = cursor.fetchall()
        profile_data['recent_accomplishments'] = [
            {
                'achievement': acc[0],
                'skill_category': acc[1], 
                'date': acc[2],
                'confidence': acc[3]
            } for acc in accomplishments
        ]
        
        conn.close()
        return profile_data
    
    def update_student_profile(self, name: str, updates: Dict[str, Any]):
        """Update student profile with new information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update basic profile
        if any(key in updates for key in ['age', 'interests', 'learning_style', 'current_level']):
            set_clause = []
            values = []
            
            for key in ['age', 'learning_style', 'current_level']:
                if key in updates:
                    set_clause.append(f"{key} = ?")
                    values.append(updates[key])
            
            if 'interests' in updates:
                set_clause.append("interests = ?")
                values.append(json.dumps(updates['interests']))
            
            set_clause.append("updated_at = CURRENT_TIMESTAMP")
            values.append(name)
            
            cursor.execute(f'''
                UPDATE student_profiles 
                SET {', '.join(set_clause)}
                WHERE name = ?
            ''', values)
        
        # Update analytics if provided
        if any(key in updates for key in ['preferred_teaching_style', 'effective_strategies', 'challenging_areas', 'motivation_triggers']):
            cursor.execute('SELECT id FROM learning_analytics WHERE student_name = ?', (name,))
            exists = cursor.fetchone()
            
            if exists:
                analytics_updates = []
                analytics_values = []
                
                for key in ['preferred_teaching_style', 'effective_strategies', 'challenging_areas', 'motivation_triggers']:
                    if key in updates:
                        if key == 'preferred_teaching_style':
                            analytics_updates.append(f"{key} = ?")
                            analytics_values.append(updates[key])
                        else:
                            analytics_updates.append(f"{key} = ?")
                            analytics_values.append(json.dumps(updates[key]))
                
                analytics_updates.append("updated_at = CURRENT_TIMESTAMP")
                analytics_values.append(name)
                
                cursor.execute(f'''
                    UPDATE learning_analytics 
                    SET {', '.join(analytics_updates)}
                    WHERE student_name = ?
                ''', analytics_values)
            else:
                cursor.execute('''
                    INSERT INTO learning_analytics 
                    (student_name, preferred_teaching_style, effective_strategies, challenging_areas, motivation_triggers)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    name,
                    updates.get('preferred_teaching_style', ''),
                    json.dumps(updates.get('effective_strategies', [])),
                    json.dumps(updates.get('challenging_areas', [])),
                    json.dumps(updates.get('motivation_triggers', []))
                ))
        
        conn.commit()
        conn.close()
    
    def add_learning_session(self, student_name: str, lesson_topic: str, agent_used: str, 
                           conversation_summary: str, effectiveness: int, notes: str = ""):
        """Record a learning session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_sessions 
            (student_name, lesson_topic, agent_used, conversation_summary, learning_effectiveness, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (student_name, lesson_topic, agent_used, conversation_summary, effectiveness, notes))
        
        conn.commit()
        conn.close()
    
    def add_accomplishment(self, student_name: str, achievement: str, skill_category: str, confidence_level: int):
        """Add a new learning accomplishment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO accomplishments (student_name, achievement, skill_category, confidence_level)
            VALUES (?, ?, ?, ?)
        ''', (student_name, achievement, skill_category, confidence_level))
        
        conn.commit()
        conn.close()
    
    def create_lesson_plan(self, student_name: str, learning_objective: str, 
                          lesson_steps: List[str], target_skills: List[str], 
                          personalization_notes: str) -> int:
        """Create a new lesson plan"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO lesson_plans 
            (student_name, learning_objective, lesson_steps, target_skills, personalization_notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            student_name, 
            learning_objective, 
            json.dumps(lesson_steps), 
            json.dumps(target_skills),
            personalization_notes
        ))
        
        lesson_plan_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return lesson_plan_id
    
    def get_current_lesson_plan(self, student_name: str) -> Optional[Dict[str, Any]]:
        """Get the current active lesson plan for a student"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM lesson_plans 
            WHERE student_name = ? AND status IN ('pending', 'in_progress')
            ORDER BY created_at DESC LIMIT 1
        ''', (student_name,))
        
        plan = cursor.fetchone()
        conn.close()
        
        if plan:
            return {
                'id': plan[0],
                'student_name': plan[1],
                'learning_objective': plan[2],
                'lesson_steps': json.loads(plan[3]),
                'target_skills': json.loads(plan[4]),
                'personalization_notes': plan[5],
                'created_at': plan[6],
                'status': plan[7]
            }
        return None
    
    def update_lesson_plan_status(self, plan_id: int, status: str):
        """Update lesson plan status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE lesson_plans SET status = ? WHERE id = ?', (status, plan_id))
        conn.commit()
        conn.close()
    
    def get_parent_dashboard(self, student_name: str) -> Dict[str, Any]:
        """Generate parent dashboard data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic profile
        profile = self.get_student_profile(student_name)
        
        # Get recent sessions
        cursor.execute('''
            SELECT lesson_topic, agent_used, learning_effectiveness, session_date, notes
            FROM learning_sessions 
            WHERE student_name = ? 
            ORDER BY session_date DESC LIMIT 10
        ''', (student_name,))
        sessions = cursor.fetchall()
        
        # Get accomplishments by category
        cursor.execute('''
            SELECT skill_category, COUNT(*), AVG(confidence_level)
            FROM accomplishments 
            WHERE student_name = ?
            GROUP BY skill_category
        ''', (student_name,))
        skill_progress = cursor.fetchall()
        
        conn.close()
        
        return {
            'student_name': student_name,
            'profile': profile,
            'recent_sessions': [
                {
                    'topic': session[0],
                    'agent': session[1],
                    'effectiveness': session[2],
                    'date': session[3],
                    'notes': session[4]
                } for session in sessions
            ],
            'skill_progress': [
                {
                    'category': skill[0],
                    'achievements_count': skill[1],
                    'average_confidence': round(skill[2], 1) if skill[2] else 0
                } for skill in skill_progress
            ]
        }

# Global database instance
db = LearningDatabase() 