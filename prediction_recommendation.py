

def recommend_action(last_value, predicted_value):
    
    if predicted_value/last_value > 1.02:
        return "BUY"
    
    
    
    if predicted_value/last_value < 0.98:
        return "SELL"
        
    else:
        return "HOLD"
        
        
        