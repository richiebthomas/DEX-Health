from flask import Flask, request, render_template
import pandas as pd
import joblib
import os
import json
from difflib import get_close_matches

app = Flask(__name__)

# Initialize model and column metadata
model = None
X_columns = None

# Load exercises data
with open('exercises.json', 'r') as f:
    exercises_data = json.load(f)

# Load GIFs data
gifs_data = pd.read_csv('gifs.csv')

def load_model():
    global model, X_columns
    if not model and os.path.exists('fitness_model.pkl'):
        model = joblib.load('fitness_model.pkl')
    if not X_columns and os.path.exists('X_columns.pkl'):
        X_columns = joblib.load('X_columns.pkl')

def find_exercise_details(exercise_name):
    """Find exercise details from the dataset"""
    # Exact match
    for exercise in exercises_data:
        if exercise['name'].lower() == exercise_name.lower():
            return exercise
    
    # Partial match
    matches = get_close_matches(exercise_name, [e['name'] for e in exercises_data], n=1, cutoff=0.6)
    if matches:
        for exercise in exercises_data:
            if exercise['name'] == matches[0]:
                return exercise
    
    return None

def find_gif(exercise_name):
    """Find GIF for the exercise"""
    gif = gifs_data[gifs_data['title'].str.lower() == exercise_name.lower()]
    if not gif.empty:
        return gif.iloc[0]['src']
    return None

def assign_workouts(schedule, exercises):
    """Assign exercises evenly across free time slots"""
    free_slots = [slot for slot in schedule if slot['status'] == 'free']
    if not free_slots:
        return schedule  # No free slots, return as is

    # Divide exercises evenly across free slots
    exercises_per_slot = len(exercises) // len(free_slots)
    remaining_exercises = len(exercises) % len(free_slots)

    ex_index = 0
    for i, slot in enumerate(free_slots):
        slot_exercises = exercises[ex_index:ex_index + exercises_per_slot + (1 if i < remaining_exercises else 0)]
        slot['exercises'] = []
        for ex in slot_exercises:
            exercise_details = find_exercise_details(ex)
            if exercise_details:
                exercise_details['gif'] = find_gif(ex)
                slot['exercises'].append(exercise_details)
        ex_index += len(slot_exercises)

    return schedule

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            load_model()
            if not model or not X_columns:
                return render_template('index.html', error="Model not trained")

            # Get user input
            input_data = {
                'Sex': request.form['sex'],
                'Age': int(request.form['age']),
                'Height': float(request.form['height']),
                'Weight': float(request.form['weight']),
                'Hypertension': request.form['hypertension'],
                'Diabetes': request.form['diabetes'],
                'BMI': float(request.form['bmi']),
                'Level': request.form['level'],
                'Fitness Goal': request.form['fitness_goal'],
                'Fitness Type': request.form['fitness_type']
            }

            # Process dynamic schedule
            schedule = []
            slot_count = int(request.form['slot_count'])
            for i in range(1, slot_count + 1):
                schedule.append({
                    'start': request.form[f'slot{i}_start'],
                    'end': request.form[f'slot{i}_end'],
                    'status': request.form[f'slot{i}_status']
                })

            # Make prediction
            input_df = pd.DataFrame([input_data])
            input_encoded = pd.get_dummies(input_df).reindex(columns=X_columns, fill_value=0)
            prediction = model.predict(input_encoded)
            
            # Split exercises and assign to schedule
            exercises = prediction[0][0].split(', ')
            assigned_schedule = assign_workouts(schedule, exercises)

            return render_template('index.html',
                exercises=exercises,
                schedule=assigned_schedule,
                equipment=prediction[0][1],
                diet=prediction[0][2],
                recommendation=prediction[0][3]
            )

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)