import datetime
import math

def checkBiogram(value, valueT, type):
    if value < -0.5:
        if valueT < -0.5 and valueT < value:
            return "Oh no! Your " + type + " biogram is low and will be even lower tomorrow... :c"
        elif valueT < -0.5 and valueT > value:
            return "Your " + type + " biogram is low and will be low tomorrow but higher than today!"
        elif valueT > -0.5:
            return "Your " + type + " biogram for tomorrow will be higher than today!"
    elif value > 0.5:
        return "Congrats! Your " + type + " biogram is high!"
    else:
        return "Your " + type + " biogram is normal."

name = input("Input your name: ")
birthYear = int(input("Input your birth year: "))
birthMonth = int(input("Input your birth month as a number: "))
birthDay = int(input("Input your birth day: "))

currentDate = datetime.datetime.now()
birthDate = datetime.datetime(birthYear, birthMonth, birthDay)
days_lived = (currentDate - birthDate).days

howLongYouLived = currentDate - datetime.datetime(birthYear, birthMonth, birthDay)
print(f"Hello {name}!")
print("You lived for", howLongYouLived.days, "days")

physical = math.sin((2 * math.pi * days_lived) / 23)
physicalTomorrow = math.sin((2 * math.pi * (days_lived + 1)) / 23)
emotional = math.sin((2 * math.pi * days_lived) / 28)
emotionalTomorrow = math.sin((2 * math.pi * (days_lived + 1)) / 28)
intellectual = math.sin((2 * math.pi * days_lived) / 33)
intellectualTomorrow = math.sin((2 * math.pi * (days_lived + 1)) / 33)


print("Physical: ", physical, checkBiogram(physical, physicalTomorrow, "physical"))
print("Emotional: ", emotional, checkBiogram(emotional, emotionalTomorrow, "emotional"))
print("Intellectual: ", intellectual, checkBiogram(intellectual, intellectualTomorrow, "intellectual"))

#made it in an hour to 2hours sth like that
