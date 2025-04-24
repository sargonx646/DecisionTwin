import sqlite3
import json
from typing import List, Dict

def init_db():
    """Initialize the SQLite database with a personas table."""
    conn = sqlite3.connect('decisionforge.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            goals TEXT,
            biases TEXT,
            tone TEXT,
            bio TEXT,
            expected_behavior TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_persona(persona: Dict):
    """Save a persona to the database, updating if it exists by name."""
    conn = sqlite3.connect('decisionforge.db')
    c = conn.cursor()
    # Check if persona exists by name
    c.execute("SELECT id FROM personas WHERE name = ?", (persona['name'],))
    existing = c.fetchone()
    
    goals_json = json.dumps(persona['goals'])
    biases_json = json.dumps(persona['biases'])
    
    if existing:
        # Update existing persona
        c.execute('''
            UPDATE personas SET
                goals = ?,
                biases = ?,
                tone = ?,
                bio = ?,
                expected_behavior = ?
            WHERE id = ?
        ''', (
            goals_json,
            biases_json,
            persona['tone'],
            persona['bio'],
            persona['expected_behavior'],
            existing[0]
        ))
    else:
        # Insert new persona
        c.execute('''
            INSERT INTO personas (name, goals, biases, tone, bio, expected_behavior)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            persona['name'],
            goals_json,
            biases_json,
            persona['tone'],
            persona['bio'],
            persona['expected_behavior']
        ))
    
    conn.commit()
    conn.close()

def update_persona(persona: Dict):
    """Update an existing persona in the database by ID."""
    conn = sqlite3.connect('decisionforge.db')
    c = conn.cursor()
    goals_json = json.dumps(persona['goals'])
    biases_json = json.dumps(persona['biases'])
    c.execute('''
        UPDATE personas SET
            name = ?,
            goals = ?,
            biases = ?,
            tone = ?,
            bio = ?,
            expected_behavior = ?
        WHERE id = ?
    ''', (
        persona['name'],
        goals_json,
        biases_json,
        persona['tone'],
        persona['bio'],
        persona['expected_behavior'],
        persona['id']
    ))
    conn.commit()
    conn.close()

def delete_persona(persona_id: int):
    """Delete a persona from the database by ID."""
    conn = sqlite3.connect('decisionforge.db')
    c = conn.cursor()
    c.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
    conn.commit()
    conn.close()

def get_all_personas() -> List[Dict]:
    """Retrieve all personas from the database."""
    conn = sqlite3.connect('decisionforge.db')
    c = conn.cursor()
    c.execute("SELECT id, name, goals, biases, tone, bio, expected_behavior FROM personas")
    rows = c.fetchall()
    personas = []
    for row in rows:
        persona = {
            "id": row[0],
            "name": row[1],
            "goals": json.loads(row[2]) if row[2] else [],
            "biases": json.loads(row[3]) if row[3] else [],
            "tone": row[4],
            "bio": row[5],
            "expected_behavior": row[6]
        }
        personas.append(persona)
    conn.close()
    return personas
