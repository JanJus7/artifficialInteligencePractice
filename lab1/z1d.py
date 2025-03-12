import datetime
import math

def check_biogram(value: float, value_tomorrow: float, category: str) -> str:
    if value < -0.5:
        if value_tomorrow < -0.5:
            if value_tomorrow < value:
                return f"Oh no! Your {category} biogram is low and will be even lower tomorrow... :c"
            return f"Your {category} biogram is low and will be low tomorrow but higher than today!"
        return f"Your {category} biogram for tomorrow will be higher than today!"
    
    if value > 0.5:
        return f"Congrats! Your {category} biogram is high!"
    
    return f"Your {category} biogram is normal."

def get_user_birthdate():
    name = input("Input your name: ")
    birth_year = int(input("Input your birth year: "))
    birth_month = int(input("Input your birth month as a number: "))
    birth_day = int(input("Input your birth day: "))
    
    return name, datetime.datetime(birth_year, birth_month, birth_day)

def calculate_biogram(birth_date: datetime.datetime):
    current_date = datetime.datetime.now()
    days_lived = (current_date - birth_date).days
    
    biograms = {
        "Physical": (math.sin((2 * math.pi * days_lived) / 23), math.sin((2 * math.pi * (days_lived + 1)) / 23)),
        "Emotional": (math.sin((2 * math.pi * days_lived) / 28), math.sin((2 * math.pi * (days_lived + 1)) / 28)),
        "Intellectual": (math.sin((2 * math.pi * days_lived) / 33), math.sin((2 * math.pi * (days_lived + 1)) / 33)),
    }
    
    return days_lived, biograms

def main():
    name, birth_date = get_user_birthdate()
    days_lived, biograms = calculate_biogram(birth_date)
    
    print(f"\nHello {name}!")
    print(f"You have lived for {days_lived} days.")
    
    for category, (today, tomorrow) in biograms.items():
        print(f"{category}: {today:.2f}", check_biogram(today, tomorrow, category.lower()))

if __name__ == "__main__":
    main()

# that is a chatGPT's refactor