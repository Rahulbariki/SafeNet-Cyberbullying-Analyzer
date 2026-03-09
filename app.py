from flask import Flask, render_template, request, jsonify
import traceback
import os

app = Flask(__name__)

# Vercel bypass for heavy 1.5GB PyTorch models
USE_MOCK = os.environ.get("VERCEL") == "1"

print("Initializing SafeNet AI Engine...")
try:
    if not USE_MOCK:
        from transformers import pipeline
        toxicity_detector = pipeline("text-classification", model="unitary/toxic-bert")
        emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        print("SafeNet Core AI Engine Ready.")
    else:
        print("Vercel Mode: Using Fast Mock Engine")
except Exception as e:
    print(f"Model load failed (likely memory limit): {e}. Falling back to Fast Mock Engine.")
    USE_MOCK = True

# Analytics Tracker
analytics = {
    "total_scanned": 128,
    "toxic_messages": 24,
    "warnings_issued": 18,
    "safe_messages": 104,
    "chart_data": [12, 19, 3, 5, 2, 3, 10, 15, 20, 10, 5, 8, 24]
}

health_score = 100

def get_coach_advice(health, message):
    if health < 40:
        return {
            "title": "⚠ Conversation Health Dropping",
            "tip": "Try expressing feedback in a respectful way. Consider giving constructive feedback instead."
        }
    elif health < 70:
        return {
            "title": "⚠ Conversation Tone Warning",
            "tip": "This conversation is becoming negative. Try using respectful language to keep the discussion constructive."
        }
    return None

def generate_polite_rewrite(message):
    msg_lower = message.lower().strip()
    if any(word in msg_lower for word in ["worthless", "useless", "terrible"]):
        return "Please try to improve this part."
    elif "nobody likes you" in msg_lower:
        return "Let’s try to work better together."
    elif "delete your account" in msg_lower:
        return "Maybe reconsider your approach."
    elif "hate" in msg_lower:
        return "I respectfully disagree with what you are saying."
    elif any(word in msg_lower for word in ["stupid", "idiot"]):
        return "I don't think this is the right way to do things."
    elif "shut up" in msg_lower:
        return "Please give me a chance to finish speaking."
    else:
        return "Could you please rephrase this to be more constructive?"

def safenet_engine(message):
    global health_score, analytics
    analytics['total_scanned'] += 1
    
    msg_l = message.lower()
    is_mock_toxic_list = any(w in msg_l for w in [
        "worthless", "useless", "terrible", "nobody likes you", 
        "delete your account", "hate", "stupid", "idiot", "shut up", 
        "kill", "loser", "ugly", "dumb", "fuck", "bitch", "shit", 
        "ass", "bastard", "crap", "whore", "cunt", "slut", "dick"
    ])

    if USE_MOCK:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from better_profanity import profanity
        
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(message)
        
        has_profanity = profanity.contains_profanity(message)
        
        is_toxic = is_mock_toxic_list or has_profanity or (vs['compound'] <= -0.4) or (vs['neg'] >= 0.4)
        
        if is_toxic:
            base_score = 0.85 if has_profanity else (0.70 + (vs['neg'] * 0.3) + abs(vs['compound'] * 0.1))
            score = min(0.99, base_score)
            if (is_mock_toxic_list or has_profanity) and score < 0.9:
                score = 0.97
        else:
            score = max(0.01, vs['neg'] * 0.5)

        label = 'toxic' if is_toxic else 'safe'
        
        # Dynamic emotion mapping from Vader
        anger_score = 0.8 if has_profanity else vs['neg'] * 0.6
        hate_score = 0.6 if has_profanity else vs['neg'] * 0.4
        neutral_score = 0.1 if has_profanity else vs['neu']
        positive_score = 0.0 if has_profanity else vs['pos']
    else:
        # Real Engine
        toxic_result = toxicity_detector(message)[0]
        score = float(toxic_result['score'])
        label = toxic_result['label'].lower()
        is_toxic = score > 0.5
        
        emotion_results = emotion_detector(message)[0]
        emotions = {e['label']: e['score'] for e in emotion_results}
        anger_score = emotions.get('anger', 0)
        hate_score = emotions.get('disgust', 0) + emotions.get('fear', 0)
        neutral_score = emotions.get('neutral', 0)
        positive_score = emotions.get('joy', 0) + emotions.get('surprise', 0)
        
    total = sum([anger_score, hate_score, neutral_score, positive_score])
    if total == 0: total = 1
    
    emotion_percentages = {
        "Anger": int((anger_score / total) * 100),
        "Hate": int((hate_score / total) * 100),
        "Neutral": int((neutral_score / total) * 100),
        "Positive": int((positive_score / total) * 100)
    }

    # Step 1: Health Updates
    if is_toxic:
        health_score -= 35
        analytics['toxic_messages'] += 1
        analytics['warnings_issued'] += 1
        analytics['chart_data'].pop(0)
        analytics['chart_data'].append(analytics['chart_data'][-1] + int(score * 10))
    else:
        total_neg = emotion_percentages['Anger'] + emotion_percentages['Hate']
        if total_neg > 40:
            health_score -= 15
        else:
            health_score += 10
            
        analytics['safe_messages'] += 1
        analytics['chart_data'].pop(0)
        analytics['chart_data'].append(max(0, analytics['chart_data'][-1] - 2))
        
    health_score = max(0, min(100, int(health_score)))
    
    # Step 2: Advice Injection
    coach_advice = get_coach_advice(health_score, message)
    suggested_rewrite = generate_polite_rewrite(message) if (is_toxic or coach_advice) else None
    
    safe_confidence = 1.0 - score if label in ['toxic', 'label_1'] else score
    if safe_confidence < 0.5: 
        safe_confidence = 0.99 

    status = "toxic" if is_toxic else "safe"
    outcome = "⚠ Hidden by SafeNet" if is_toxic else "✅ Message delivered successfully"
    shield_action = "Comment blocked" if is_toxic else "Message allowed"
    warning = "SafeNet Warning:\nThis message may harm someone." if is_toxic else "No warning needed."

    return {
        "status": status,
        "message_outcome": outcome,
        "confidence": score if is_toxic else safe_confidence,
        "emotions": emotion_percentages,
        "warning": warning,
        "suggested_rewrite": suggested_rewrite,
        "health_score": health_score,
        "coach_advice": coach_advice,
        "shield_action": shield_action,
        "analytics_update": analytics
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        msg = data.get('message', '')
        if not msg:
            return jsonify({"status": "safe", "health_score": health_score})
        
        return jsonify(safenet_engine(msg))
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        return jsonify({"status": "error"})

@app.route('/dashboard', methods=['GET'])
def get_dashboard():
    return jsonify(analytics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
