import math
import random
import matplotlib.pyplot as plt

g = 9.81
v0 = 50
h = 100

def calculateDistance(angle):
    angleRad = math.radians(angle)
    distance = (v0 * math.sin(angleRad) + math.sqrt((v0 * math.sin(angleRad))**2 + 2 * g * h)) * (v0 * math.cos(angleRad)) / g
    return distance

def plotTrajectory(angle, distance):
    angleRad = math.radians(angle)
    xValues = [i for i in range(0, int(distance) + 1)]
    yValues = [(-g / (2 * v0**2 * math.cos(angleRad)**2)) * x**2 + (math.tan(angleRad)) * x + h for x in xValues]
    
    plt.figure()
    plt.plot(xValues, yValues, color='blue')
    plt.title('Projectile Trajectory')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.grid(True)
    plt.savefig('trajectory.png')
    plt.show()

targetDistance = random.randint(50, 340)
print(f"Target is at {targetDistance} meters.")

attempts = 0
while True:
    angle = float(input("Enter the angle (in degrees): "))
    distance = calculateDistance(angle)
    attempts += 1
    
    if abs(distance - targetDistance) <= 5:
        print("Target hit!")
        print(f"The shot flew {distance:.2f} meters.")
        print(f"Total attempts: {attempts}")
        plotTrajectory(angle, distance)
        break
    else:
        print(f"Missed. The shot flew {distance:.2f} meters. Try again.")