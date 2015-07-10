#Random coordinate generator
#For individual huamn identification
#I: h + k + l = 2n
#F: h + k = 2n, k + l = 2n, h + l = 2n
#A: k + l = 2n
#B: h + l = 2n
#C: h + k = 2n
#R: -h + k + l = 3n
#P: Anything goes

import numpy as np
import random
import math

#Generates a random even number
def genEven():
    x = 2*random.randrange(-5,6,1)
    return x

#Generates a random odd number
def genOdd():
    x = 2*random.randrange(-5,5,1)+1
    return x

#Generates a random integer within range
def genInt():
    x = random.randrange(-10,11,1)
    return x

#Fixes -coordinates that are out of bounds
def fix_outOfBounds(h, k, l):
    if h < -10: h += 3
    elif h > 10: h -= 3
    
    if k < -10: k += 3
    elif k > 10: k -= 3
    
    if l < -10: l += 3
    elif l > 10: l -= 3
    
    return h, k, l

def genCoords():
    I_coords = np.empty([6000,3]); F_coords = np.empty([6000,3]); A_coords = np.empty([6000,3]);
    B_coords = np.empty([6000,3]); C_coords = np.empty([6000,3]); R_coords = np.empty([6000,3])
    P_coords = np.empty([6000,3])
    
    #Generate I coordinates
    for i in range(6000):
        #2 possibilities: all even, or 1 value is even
        j = random.random()
        if (j <= .2874109263): #matches the probability of their being all even (1331/4631)
            h = genEven(); k = genEven(); l = genEven()
            I_coords[i] = [h, k, l]
        else: #1 is even
            which_one = random.random()
            if (which_one >= .6666666667):
                h = genEven(); k = genOdd(); l = genOdd()
                I_coords[i] = [h, k, l]
            elif (which_one >= .3333333333):
                h = genOdd(); k = genEven(); l = genOdd()
                I_coords[i] = [h, k, l]
            else:
                h = genOdd(); k = genOdd(); l = genEven()
                I_coords[i] = [h, k, l]
    
    #Generate F coordinates
    for i in range(6000):
        #2 possibilities: all even, or all odd
        j = random.random()
        if (j <= .5709995710): #prob(all even) = 1331/2331
            h = genEven(); k = genEven(); l = genEven()
            F_coords[i] = [h, k, l]
        else:
            h = genOdd(); k = genOdd(); l = genOdd()
            F_coords[i] = [h, k, l]
            
    #Generate A coordinates
    for i in range(6000):
        #k and l are both even or both odd
        j = random.random()
        if (j <= .5475113122): #prob(both even) = 121/221
            h = genInt(); k = genEven(); l = genEven()
            A_coords[i] = [h,k,l]
        else:
            h = genInt(); k = genOdd(); l = genOdd()
            A_coords[i] = [h,k,l]
    
    #Generate B coordinates
    for i in range(6000):
        #h and l are both even or both odd
        j = random.random()
        if (j <= .5475113122): #prob(both even) = 121/221
            h = genEven(); k = genInt(); l = genEven()
            B_coords[i] = [h,k,l]
        else:
            h = genOdd(); k = genInt(); l = genOdd()
            B_coords[i] = [h,k,l]    
            
    #Generate C coordinates
    for i in range(6000):
        #h and k are both even or both odd
        j = random.random()
        if (j <= .5475113122): 
            h = genEven(); k = genEven(); l = genInt()
            C_coords[i] = [h,k,l]
        else:
            h = genOdd(); k = genOdd(); l = genInt()
            C_coords[i] = [h,k,l]
            
    #Generate R coordinates
    for i in range(6000):
        remainders = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1], [1, 1, 0], [1, 0, 1], [2, 2, 0], [2, 0, 2], [2, 1, 1], [1, 2, 2]])
        h = random.randrange(-12, 10, 3)
        k = random.randrange(-12, 10, 3)
        l = random.randrange(-12, 10, 3)
    
        j = random.randint(0, 8)
        h += remainders[j][0]; k += remainders[j][1]; l += remainders[j][2]
    
        #Fixing -6
        h, k, l, = fix_outOfBounds(h, k, l)
        R_coords[i] = [h,k,l]
        
    #Generate P coordinates
    for i in range(6000):
        P_coords[i] = [genInt(), genInt(), genInt()]    
        
    allCoords = np.vstack([I_coords, F_coords, A_coords, B_coords, C_coords, R_coords, P_coords]) #massive array
    return allCoords

#Formats training data
def buildTrain(coords):
    myList = []
    for i in range(len(coords)):
        myList.append([coords[i][0], coords[i][1], coords[i][2], \
                       coords[i][0]%2, coords[i][1]%2, coords[i][2]%2, \
                       coords[i][0]%3, coords[i][1]%3, coords[i][2]%3, \
                       math.floor(i/6000)]) #I = 0, F = 1, A = 2, B = 3, C = 4, R = 5, P = 6
        
    myArr = np.array(myList)
    return myArr

#Formats testing data
def buildTest(coords):
    myList = []
    for i in range(len(coords)): #No answer this time
        myList.append([coords[i][0], coords[i][1], coords[i][2], \
                       coords[i][0]%2, coords[i][1]%2, coords[i][2]%2, \
                       coords[i][0]%3, coords[i][1]%3, coords[i][2]%3])
        
    myArr = np.array(myList)
    return myArr
    
