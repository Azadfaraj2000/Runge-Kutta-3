# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:42:56 2021
@author: Azad Faraj, ID: 30164788
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# %%

def stepsize(interval,N):
    """
    
    Parameters
    ----------
    interval : 2 element list of floats
        list containing start and end x-values.
    N : integer
        number of steps.

    Returns
    -------
    h : float
        Grid step size.
    x0 : float
        start x-value.
    xend : float
        end x-value.
        
    """
    assert len(interval) ==2,"interval must be a 2 element array"
    x0 = interval[0]; xend = interval[-1]; 
    h= (xend-x0)/N
    return h, x0, xend

def rk3(A, bvector, y0, interval, N):
    """
    
    Standard explict third order Runge-Kutta method

    Parameters
    ----------
    A : matrix
        an n × n matrix with constant coefficient.
    bvector : function
        takes the form b = bvector(x). Vector b depends only on the independent
        variable x.
    y0 : array of floats
        n-vector of initial data.
    interval : 2 element list of floats
        2 element list giving the start and end values.
    N : integer
        number of steps.

    Returns
    -------
    x : matrix of floats
        x values.
    y : matrix of floats
        y values.

    """
    assert N>1, "Number must be greater than 1"
    assert (int(N) == N), "Number of steps must be an integer"
    assert A.shape[0] == A.shape[1], "A must be square"
    assert len(A)==len(y0), "Size of A and y0 must match"
    assert ((not np.any(np.isnan(y0)))and np.all(np.isfinite(y0))),"Enter valid y0"
    h, x0, xend = stepsize(interval,N) #get stepsize, start and end x values
    x = np.linspace(x0,xend ,N+1) #create linearally spaced grid
    y = [y0] #allocate memory for y values
    yn=y0
    for n in range(N):
        xn = x[n] #current step
        #calculate derivative approximations
        y1 = yn+h*(A.dot(yn) + bvector(xn))
        y2 = (yn + (1/3)*y1 + (1/3)*h*(A.dot(y1) + bvector(xn+h)))*(3/4)
        #calculate new y estimation
        yn= (yn + 2*y2 + 2*h*(A.dot(y2) + bvector(xn+h)))/3;
        #store y estimation
        y.append(yn)
    #return x,y values
    return x,np.asarray(y).T

def dirk3(A, bvector, y0, interval, N):
    """
    
    Third order accurate Diagonally Implicit Runge-Kutta method
    
    Parameters
    ----------
    A : matrix
        an n × n matrix with constant coefficient.
    bvector : function
        takes the form b = bvector(x). Vector b depends only on the independent
        variable x.
    y0 : array of floats
        n-vector of initial data.
    interval : 2 element list of floats
        2 element list giving the start and end values.
    N : integer
        number of steps.

    Returns
    -------
    x : matrix of floats
        x values.
    y : matrix of floats
        y values.

    """
    assert N>1, "Number must be greater than 1"
    assert (int(N) == N), "Number of steps must be an integer"
    assert A.shape[0] == A.shape[1], "A must be square"
    assert len(A)==len(y0), "Size of A and y0 must match"
    assert ((not np.any(np.isnan(y0)))and np.all(np.isfinite(y0))),"Enter valid y0"
    h, x0, xend = stepsize(interval,N) #get stepsize, start and end x values
    x = np.linspace(x0,xend ,N+1)
    y = [y0]
    yn=y0
    
    # coefficients
    mu = 1/2 * (1-(1/np.sqrt(3))) #μ
    nu = 1/2 * (np.sqrt(3)-1)     #ν
    gamma = 3 /(2*(3+np.sqrt(3))) #γ
    lam = 3*(1+np.sqrt(3)) / (2*(3+np.sqrt(3)))#λ; name 'lambda' already taken,
                                               #so lam is used instead
    I = np.identity(len(A)) #identity matrix 
    for n in range(N):
        xn = x[n] #current step
        #calculate derivative approximations
        a = (I-h*mu*A)
        b = (yn + h*mu*(bvector(xn+h*mu)) )
        y1 = np.linalg.solve(a,b)
        b = y1+h*nu*(A.dot(y1)+bvector(xn+h*mu))+h*mu*(bvector(xn+h*nu+2*h*mu)) 
        y2 = np.linalg.solve(a,b)
        ##calculate new y estimation
        yn= (yn + (lam*y2)/((1-lam)) 
             + h*gamma*(A.dot(y2)+bvector(xn+h*nu+2*h*mu))/(1-lam) )*(1-lam)
        #store y estimation
        y.append(yn)
    #return x,y values
    return x,np.asarray(y).T

#%%

"""
-------------------------------------------------------------------------------
Task 3 - Moderately stiff case
-------------------------------------------------------------------------------
"""


a1 = 1000; a2 = 1 #explicit values
A = np.array([[-a1, 0],
              [a1, -a2]])
y0=[1,0] #initial data
interval = [0,0.1] #interval x ∈ [0, 0.1]
bvector = lambda x: 0 #vector is trivial
#------------------------------------------------------------------------------
#1-Norm Error
#------------------------------------------------------------------------------

h_array=[] #allocate memory
normError1_array_rk3=[] #allocate memory
normError1_array_dirk3=[] #allocate memory

for k in range(1,11):  #k=1,2,3,...,10
    """Computes the 1-norm of the relative error for both RK3 & DIRK3"""
    assert k>=1, "k must be greater or equal to 1"
    assert (int(k) == k), "k must be an integer"
    N=40*k
    x, y_rk3 = rk3(A, bvector, y0, interval, N)
    y1_rk3= y_rk3[0]; y2_rk3 = y_rk3[1]
    
    x_dirk3, y_dirk3 = dirk3(A, bvector, y0, interval, N)
    y1_dirk3= y_dirk3[0]; y2_dirk3 = y_dirk3[1]
    
    yexact1 = np.exp(-a1*x)
    yexact2 = (a1/(a1-a2)) * (np.exp(-a2*x) -np.exp(-a1*x))
    
    normError1_rk3=0
    normError1_dirk3=0
    for j in range(1,N+1):
        h = stepsize(interval,N)[0]
        normError1_rk3=normError1_rk3+abs(h*((y2_rk3[j]-yexact2[j])/yexact2[j]))
        normError1_dirk3=normError1_dirk3+abs(h*((y2_dirk3[j]-yexact2[j])/yexact2[j]))
    h_array.append(h)
    
    normError1_array_rk3.append(normError1_rk3)
    normError1_array_dirk3.append(normError1_dirk3)

#----rk3----
#----3rd-order Polynomial fit----
coefficients = np.polyfit(np.log10(h_array), np.log10(normError1_array_rk3), 1)
polynomial = np.poly1d(coefficients)
log10_y_fit = polynomial(np.log10(h_array))

#----plot----
plt.plot(h_array, 10**log10_y_fit, linestyle=":",alpha=0.5, label = r'$∝h^3$')  
plt.scatter(h_array, normError1_array_rk3, s = 9, color = 'r', label = 'RK3')

plt.suptitle('Convergence rate - Moderately Stiff Case - RK3')
plt.yscale('log'); plt.xscale('log')
plt.xlabel(r"$h$",fontsize=15); plt.ylabel(r'$||Error||_{1}$',fontsize=15)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.gca().invert_xaxis() #reverse x-axis
plt.legend()
plt.show()

#----dirk3----
#----3rd-order Polynomial fit----
coefficients = np.polyfit(np.log10(h_array),np.log10(normError1_array_dirk3),1)
polynomial = np.poly1d(coefficients)
log10_y_fit = polynomial(np.log10(h_array))
#----plot----
plt.plot(h_array, 10**log10_y_fit, linestyle=":",alpha=0.5, label = r'$∝h^3$')
plt.scatter(h_array, normError1_array_dirk3, s = 9, color='r', label ='DIRK3')
plt.suptitle('Convergence rate - Moderately Stiff Case - DIRK3')
plt.xlabel("h"); plt.ylabel("1-Norm Error")
plt.yscale('log'); plt.xscale('log')
plt.xlabel(r"$h$",fontsize=15); plt.ylabel(r'$||Error||_{1}$',fontsize=15)
plt.gca().invert_xaxis() #reverse x-axis
plt.legend()
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()


#------------------------------------------------------------------------------
#y1 & y2 against x
#------------------------------------------------------------------------------

N=400

#rk3
x, y = rk3(A, bvector, y0, interval, N)
y1= y[0]; y2 = y[1]

yexact1 = np.exp(-a1*x)
yexact2 = (a1/(a1-a2)) * (np.exp(-a2*x) -np.exp(-a1*x))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('RK3')

#subplot y1
ax1.semilogy(x, y1,label ="N=400 (highest resolution)")
ax1.plot(x,yexact1,label ="Exact solution", linestyle=":")
ax1.set_xlabel("x"); ax1.set_ylabel(r'$y_{1}$')
ax1.minorticks_on()
ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#subplot y2
ax2.plot(x, y2)
ax2.plot(x,yexact2, linestyle=":")
ax2.set_xlabel("x"); ax2.set_ylabel(r'$y_{2}$')
ax2.minorticks_on()
ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

fig.tight_layout()
fig.legend(prop={'size': 10})
plt.show()

#dirk3
x, y = dirk3(A, bvector, y0, interval, N)
y1= y[0]; y2 = y[1]

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('DIRK3')

#subplot y1
ax1.semilogy(x, y1,label ="N=400 (highest resolution)")
ax1.plot(x,yexact1,label ="Exact solution", linestyle=":")
ax1.set_xlabel("x"); ax1.set_ylabel(r'$y_{1}$')
ax1.minorticks_on()
ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

#subplot y2
ax2.plot(x, y2)
ax2.plot(x,yexact2, linestyle=":")
ax2.set_xlabel("x"); ax2.set_ylabel(r'$y_{2}$')
ax2.minorticks_on()
ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

fig.tight_layout()
fig.legend(prop={'size': 10})
plt.show()


# %%

"""
-------------------------------------------------------------------------------
Task 4 - Stiff Case
-------------------------------------------------------------------------------
"""


A = np.array([[-1,      0,      0],
              [-99,   -100,     0],
              [-10098, 9900, -10000]])



bvector = lambda x: np.array([np.cos(10*x)-10*np.sin(10*x),
                              199*np.cos(10*x)-10*np.sin(10*x),
                              208*np.cos(10*x)+10000*np.sin(10*x)])
y0=[0, 1, 0] # initial data
interval = [0, 1] #interval x ∈ [0, 1]


#------------------------------------------------------------------------------
#1-Norm Error
#------------------------------------------------------------------------------


h_array=[] #allocate memory
normError1_array=[] #allocate memory
for k in range(4,17):  #k=4,5,6,...,16
    """Computes the 1-norm of the relative error for DIRK3 only"""
    assert k>=1, "k must be greater or equal to 1"
    assert (int(k) == k), "k must be an integer"
    N=200*k
    x, y = dirk3(A, bvector, y0, interval, N)
    y1= y[0]; y2 = y[1]; y3 = y[2]
    yexact1 = np.cos(10*x) - np.exp(-x)
    yexact2 = np.cos(10*x) + np.exp(-x) - np.exp(-100*x)
    yexact3 = np.sin(10*x) + 2*np.exp(-x) - np.exp(-100*x) - np.exp(-10000*x)
    normError1=0
    for j in range(1,N+1):
        h = stepsize(interval,N)[0]
        normError1 = normError1 + abs(h * ( (y3[j]-yexact3[j])/yexact3[j] ))
    h_array.append(h)
    normError1_array.append(normError1)

#----polynomial fit----
h_array=np.array(h_array)
normError1_array=np.array(normError1_array)
fit = np.polyfit(h_array, normError1_array, 3)
a = fit[0]; b = fit[1]; c = fit[2]; d=fit[3] #coefficients
fit_equation = a * np.power(h_array, 3) + b * np.square(h_array) + c*h_array +d


#----plot----
plt.plot(h_array, fit_equation,alpha = 0.5, label = r'$∝h^3$',linestyle=":")
plt.scatter(h_array, normError1_array, s = 7, color = 'r', label = 'DIRK3')
plt.suptitle('Convergence rate - Stiff Case - DIRK3')
plt.xlabel(r'$h$',fontsize=15)
plt.ylabel(r'$||Error||_{1}$',fontsize=15)
plt.legend()
plt.minorticks_on()
plt.gca().invert_xaxis() #reverse x-axis
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()


#------------------------------------------------------------------------------
#y1, y2 & y3 against x
#------------------------------------------------------------------------------


#----rk3----
N=3200
x, y = rk3(A, bvector, y0, interval, N)
y1= y[0]; y2 = y[1]; y3 = y[2]


yexact1 = np.cos(10*x) - np.exp(-x)
yexact2 = np.cos(10*x) + np.exp(-x) - np.exp(-100*x)
yexact3 = np.sin(10*x) + 2*np.exp(-x) - np.exp(-100*x) - np.exp(-10000*x)

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('RK3')

#subplot y1
ax1.plot(x, y1,label ="N=3200 (highest resolution)")
ax1.plot(x,yexact1,label ="Exact solution", linestyle=":")
#formating and labelling
ax1.set_ylabel(r'$y_{1}$',fontsize=12)
ax1.minorticks_on()
ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

#subplot y2
ax2.plot(x, y2)
ax2.plot(x,yexact2, linestyle=":")
#formating and labelling
ax2.set_ylabel(r'$y_{2}$',fontsize=12)
ax2.minorticks_on()
ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

#subplot y3
#ax3.plot(x, y3)
ax3.plot(x,yexact3, linestyle=":")
ax3.set_xlabel("x",fontsize=12); ax3.set_ylabel(r'$y_{3}$',fontsize=12)
#formating and labelling
ax3.minorticks_on()
ax3.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
fig.tight_layout(); fig.legend(prop={'size': 10})
plt.show()


#----dirk3----
N=3200
x, y = dirk3(A, bvector, y0, interval, N)
y1= y[0]; y2 = y[1]; y3 = y[2]

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('DIRK3')

#subplot y1
ax1.plot(x, y1,label ="N=3200 (highest resolution)")
ax1.plot(x,yexact1,label ="Exact solution", linestyle=":")
#formating and labelling
ax1.set_ylabel(r'$y_{1}$',fontsize=12)
ax1.minorticks_on()
ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

#subplot y2
ax2.plot(x, y2)
ax2.plot(x,yexact2, linestyle=":")
#formating and labelling
ax2.set_ylabel(r'$y_{2}$',fontsize=12)
ax2.minorticks_on()
ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

#subplot y3
ax3.plot(x, y3)
ax3.plot(x,yexact3, linestyle=":")
#formating and labelling
ax3.set_xlabel("x",fontsize=12); ax3.set_ylabel(r'$y_{3}$',fontsize=12)
ax3.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
fig.tight_layout(); ax3.minorticks_on(); fig.legend(prop={'size': 10})
plt.show()
